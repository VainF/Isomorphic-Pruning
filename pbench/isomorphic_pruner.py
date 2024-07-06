import torch_pruning as tp
import torch
import torch.nn as nn
import typing 
from torch_pruning.pruner import function
import timm
from fastcore.basics import patch_to
import torch.nn.functional as F


class IsomorphicPruner(tp.algorithms.MetaPruner):
    def __init__(
        self,
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        reg=1e-4, # regularization coefficient
        alpha=4, # regularization scaling factor, [2^0, 2^alpha]
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratio, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = tp.algorithms.scheduler.linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to
        threshold = 0.01,

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [torch.nn.Conv2d, torch.nn.Linear],  # root module for each group
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel groups for layers
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
    ):
        super(IsomorphicPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            forward_fn=forward_fn,
            output_transform=output_transform,
            
            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict
        )

        self.reg = reg
        self.alpha = alpha
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))
        self.cnt = 0
        self.threshold = threshold

    def step(self, interactive=False)-> typing.Union[typing.Generator, None]:
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        if interactive: # yield groups for interactive pruning
            return pruning_method() 
        else:
            for group in pruning_method():
                group.prune()
        self._groups = list(self.DG.get_all_groups(root_module_types=self.root_module_types, ignored_layers=self.ignored_layers))

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            return
        
        ##############################################
        # 1. Pre-compute importance for each substructure 
        # and group them by isomorphism
        ##############################################
        global_head_importance = {} # for attn head pruning
        all_groups = []
        isomorphic_groups = {}

        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types):
            if self._check_pruning_ratio(group):    
                group = self._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group) # raw importance score
                group_size = len(imp) // ch_groups
                if imp is None: continue
                if ch_groups > 1:
                    # average importance across groups. For example:
                    # imp = [1, 2, 3, 4, 5, 6] with ch_groups=2
                    # We have two groups [1,2,3] and [4,5,6]
                    # the average importance is [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0) 
                else:
                    # no grouping
                    dim_imp = imp

                # Graphs will always be generated in the forwarding order
                group_tag = ""
                for dep, _ in group:
                    # to check the isomorphism, all connected nodes should have the same tag, 
                    # i.e., the same module type & pruning dim
                    tag_source = "%s_%s"%(type(dep.source.module), "out" if self.DG.is_out_channel_pruning_fn(dep.handler) else "in")
                    tag_target = "%s_%s"%(type(dep.target.module), "out" if self.DG.is_out_channel_pruning_fn(dep.handler) else "in")
                    group_tag += "%s_%s"%(tag_source, tag_target)
                if group_tag not in isomorphic_groups:
                    # New isomorphic group
                    isomorphic_groups[group_tag] = []
                isomorphic_groups[group_tag].append((group, ch_groups, group_size, dim_imp))

                # pre-compute head importance for attn heads
                _is_attn, qkv_layers = self._is_attn_group(group)
                if _is_attn and self.prune_num_heads and self.get_target_head_pruning_ratio(qkv_layers[0])>0:
                    # average importance of each group. For example:
                    # the importance score of the group
                    # imp = [1, 2, 3, 4, 5, 6] with num_heads=2
                    # Note: head1 = [1, 2, 3], head2 = [4, 5, 6]
                    # the average importance is [(1+2+3)/3, (4+5+6)/3] = [2, 5]
                    head_imp = imp.view(ch_groups, -1).mean(1) # average importance by head.
                    global_head_importance[group] = (qkv_layers, head_imp)

        if len(isomorphic_groups) == 0 and len(global_head_importance)==0:
            return
        
        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################
        
        # Find the threshold for global pruning
        for group_type, (group_tag, global_importance) in enumerate(isomorphic_groups.items()):
            print("Prune Type {}, Tag {}".format(group_type, group_tag))
            if len(global_importance)>0:
                concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
                target_pruning_ratio = self.get_target_pruning_ratio(global_importance[0][0][0].dep.target.module)
                #self.per_step_pruning_ratio[self.current_step]
                print(target_pruning_ratio)
                n_pruned = len(concat_imp) - int(
                    len(concat_imp) *
                    (1 - target_pruning_ratio)
                )
                if n_pruned>0:
                    topk_imp, _ = torch.topk(concat_imp, k=n_pruned, largest=False)
                    thres = topk_imp[-1]

            # Find the threshold for head pruning
            if len(global_head_importance)>0:
                concat_head_imp = torch.cat([local_imp[-1] for local_imp in global_head_importance.values()], dim=0)
                target_head_pruning_ratio = self.per_step_head_pruning_ratio[self.current_step]
                n_heads_removed = len(concat_head_imp) - int(
                    self.initial_total_heads *
                    (1 - target_head_pruning_ratio)
                )
                if n_heads_removed>0:
                    topk_head_imp, _ = torch.topk(concat_head_imp, k=n_heads_removed, largest=False)
                    head_thres = topk_head_imp[-1]
            
            ##############################################
            # 3. Prune
            ##############################################
            for group, ch_groups, group_size, imp in global_importance:
                print("Prune Type {}".format(group_type))
                print(group)
                module = group[0].dep.target.module
                pruning_fn = group[0].dep.handler
                get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.DG.get_in_channels
                
                # Prune feature dims/channels
                pruning_indices = []
                if len(global_importance)>0 and n_pruned>0:
                    if ch_groups > 1: # re-compute importance for each channel group if channel grouping is enabled
                        n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
                        if n_pruned_per_group>0:
                            if self.round_to:
                                n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                            if n_pruned_per_group >= group_size: n_pruned_per_group-=1 # keep at least one channel
                            _is_attn, _ = self._is_attn_group(group)
                            if not _is_attn or self.prune_head_dims==True:
                                raw_imp = self.estimate_importance(group) # re-compute importance
                                for chg in range(ch_groups): # determine pruning indices for each channel group independently
                                    sub_group_imp = raw_imp[chg*group_size: (chg+1)*group_size]
                                    sub_imp_argsort = torch.argsort(sub_group_imp)
                                    sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group]+chg*group_size
                                    pruning_indices.append(sub_pruning_idxs)
                    else:
                        _pruning_indices = (imp <= thres).nonzero().view(-1)
                        if len(_pruning_indices)==len(imp):
                            # keep at least one channel
                            new_thres = torch.max(imp)
                            _pruning_indices = (imp < new_thres).nonzero().view(-1)
                        imp_argsort = torch.argsort(imp)
                        if len(_pruning_indices)>0 and self.round_to: 
                            n_pruned = len(_pruning_indices)
                            current_channels = get_channel_fn(module)
                            n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                            _pruning_indices = imp_argsort[:n_pruned]
                        pruning_indices.append(_pruning_indices)
                            
                # Prune heads
                
                if len(global_head_importance)>0 and n_heads_removed>0:
                    if group in global_head_importance:
                        qkv_layers, head_imp = global_head_importance[group]
                        head_pruning_indices = (head_imp <= head_thres).nonzero().view(-1)
                        head_keep_indices = (head_imp > head_thres).nonzero().view(-1)
                        if len(head_pruning_indices)==len(head_imp):
                            # keep at least the most important head
                            new_thres = torch.max(head_imp)
                            head_pruning_indices = (head_imp < new_thres).nonzero().view(-1)
                            head_keep_indices = (head_imp >= new_thres).nonzero().view(-1)
                        if len(head_pruning_indices)>0:
                            for head_id in head_pruning_indices:
                                pruning_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=head_imp.device) )
                        for qkv_layer in qkv_layers:
                            self.num_heads[qkv_layer] -= len(head_pruning_indices) # update num heads after pruning

                        for m in self.model.modules():
                            if isinstance(m, timm.models.swin_transformer.WindowAttention):
                                if m.qkv in qkv_layers:
                                    m.relative_position_bias_table = nn.Parameter(
                                        m.relative_position_bias_table[:, head_keep_indices], requires_grad=m.relative_position_bias_table.requires_grad
                                    )
                
                if len(pruning_indices)==0: continue
                pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
                # create pruning group
                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_indices)

                #if self.DG.check_pruning_group(group):
                yield group 
