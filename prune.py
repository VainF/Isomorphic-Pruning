import os, sys
import torch
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import timm
import torch_pruning as tp

import peval
peval.forward_patch.patch_timm_forward() # patch timm.forward() to support pruning

from tqdm import tqdm
import argparse

import torchvision as tv

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ResNet Pruning')
    
    # Model
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    parser.add_argument('--ckpt', default=None, type=str, help='ckpt path')
    parser.add_argument('--is-torchvision', default=False, action='store_true', help='use torchvision model')

    # Data
    parser.add_argument('--data-path', default='data/imagenet', type=str, help='data path')
    parser.add_argument('--taylor-batchs', default=50, type=int, help='number of batchs for taylor criterion')
    parser.add_argument('--no-imagenet-mean-std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--train-batch-size', default=64, type=int, help='train batch size')
    parser.add_argument('--val-batch-size', default=128, type=int, help='val batch size')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation mode', choices=['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'])
    parser.add_argument('--val-resize', default=256, type=int, help='resize size')
    parser.add_argument('--input-size', default=224, type=int, help='input size')
    
    # Pruning
    parser.add_argument('--pruning-ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--head-pruning-ratio', default=0.0, type=float, help='head pruning ratio')
    parser.add_argument('--head-dim-pruning-ratio', default=None, type=float, help='head dim pruning ratio')
    parser.add_argument('--pruning-type', default='taylor', type=str, help='pruning type', choices=['random', 'taylor', 'l2', 'l1', 'hessian'])
    parser.add_argument('--test-accuracy', default=False, action='store_true', help='test accuracy')
    parser.add_argument('--global-pruning', default=False, action='store_true', help='global pruning')
    parser.add_argument('--save-as', default=None, type=str, help='save the pruned model')
    parser.add_argument('--round-to', default=2, type=int, help='round to')
    parser.add_argument('--normalizer', default='mean', type=str)
    parser.add_argument('--remove-layers', default=0, type=int)
    parser.add_argument('--no-isomorphic', default=False, action='store_true')
    parser.add_argument('--stochastic-depth-prob', default=0.5, type=float)

    # Drop
    parser.add_argument('--drop', default=0.0, type=float, help='drop rate')
    parser.add_argument('--drop-path', default=0.0, type=float, help='drop path rate')

    args = parser.parse_args()
    return args

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=True, interpolation='bicubic', val_resize=256):
    """The imagenet_root should contain train and val folders.
    """
    interpolation = getattr(T.InterpolationMode, interpolation.upper())

    print('Parsing dataset...')
    train_dst = ImageFolder(os.path.join(imagenet_root, 'train'), 
                            transform=peval.data.presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=val_resize,
                                interpolation=interpolation,
                            )
    )
    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), 
                          transform=peval.data.presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=val_resize,
                                interpolation=interpolation,
                            )
    )
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(val_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example_inputs = torch.randn(1,3,224,224)
    train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, use_imagenet_mean_std= not args.no_imagenet_mean_std, val_resize=args.val_resize, interpolation=args.interpolation)

    # Pruning
    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == 'l2':
        imp = tp.importance.GroupNormImportance(p=2)
    elif args.pruning_type == 'l1':
        imp = tp.importance.GroupNormImportance(p=1)
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()
    else: raise NotImplementedError

    # Load the model
    if args.is_torchvision:
        import torchvision
        print(f"Loading torchvision model {args.model}...")
        model = torchvision.models.__dict__[args.model](pretrained=True).eval()
    else:
        print(f"Loading timm model {args.model}...")
        model = timm.create_model(args.model, pretrained=True, drop_rate=args.drop, drop_path_rate=args.drop_path).eval()
    
    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}...")
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    
    model.to(device)
    input_size = [3, args.input_size, args.input_size]
    example_inputs = torch.randn(1, *input_size).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    print("Pruning %s..."%args.model)
    ignored_layers = []
    num_heads = {}
    for name, m in model.named_modules():
        if (('fc' in name) or ('classifier' in name) or ('head' in name) or ('linear' in name) ) and isinstance(m, nn.Linear) and m.out_features==1000:
            print("Ignoring %s"%name)
            ignored_layers.append(m) # only prune the internal layers of FFN & Attention
        
        if isinstance(m, timm.models.swin_transformer.WindowAttention):
            print("Ignoring %s.relative_position_bias_table"%name)
            num_heads[m.qkv] = m.num_heads 
            ignored_layers.append(m.relative_position_bias_table)
        
        if isinstance(m, timm.models.vision_transformer.Attention):
            num_heads[m.qkv] = m.num_heads

    if args.test_accuracy:
        print("Testing accuracy of the original model...")
        acc_ori, loss_ori = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

    pruning_ratio_dict = {}
    if args.head_dim_pruning_ratio is not None:
        for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                pruning_ratio_dict[m] = args.head_dim_pruning_ratio

    PrunerCLS = tp.pruner.MetaPruner if args.no_isomorphic else peval.isomorphic_pruner.IsomorphicPruner
    pruner = PrunerCLS(
        model, 
        example_inputs=example_inputs,
        global_pruning=args.global_pruning, # If False, a uniform sparsity will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=args.pruning_ratio, # target sparsity
        pruning_ratio_dict=pruning_ratio_dict,
        ignored_layers=ignored_layers, # ignored layers
        num_heads=num_heads, # number of heads in self attention
        prune_head_dims=True, # prune head_dims
        prune_num_heads=True, # prune num_heads
        head_pruning_ratio=args.head_pruning_ratio, # target sparsity for heads
        customized_pruners=peval.extension.EXTENDED_PRUNERS, # customized pruners 
        round_to=args.round_to, # round to
    )

    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        model.zero_grad()
        if isinstance(imp, tp.importance.GroupHessianImportance):
            imp.zero_grad() # clear the accumulated gradients
        print("Accumulating gradients for pruning...")
        for k, (imgs, lbls) in enumerate(tqdm(train_loader)):
            if k>=args.taylor_batchs: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs)
            if isinstance(imp, tp.importance.GroupHessianImportance): # per-sample gradients for hessian
                loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model) # accumulate gradients
            elif isinstance(imp, tp.importance.GroupTaylorImportance): # batch gradients for first-order taylor
                loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()
    
    print("========Before pruning========")
    print(model)   

    # Prune
    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()
    
    # Modify static attributes
    for m in model.modules():
        if isinstance(m, timm.models.swin_transformer.WindowAttention):
            m.num_heads = pruner.num_heads[m.qkv]
            head_dim = m.qkv.out_features // (3 * m.num_heads)
            m.scale = head_dim ** -0.5
            print("Num heads: %d, head dim: %d"%(m.num_heads, head_dim))
            print()

        if isinstance(m, timm.models.vision_transformer.Attention):
            m.num_heads = pruner.num_heads[m.qkv]
            m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            m.scale = m.head_dim ** -0.5
            print("Num heads: %d, head dim: %d"%(m.num_heads, m.head_dim))
            print()

    # Remove layers
    if args.remove_layers>0:
        if isinstance(model, tv.models.convnext.ConvNeXt):
            #target_blocks = [model.features[2]] 
            block_imp = []
            for convnext_block in model.features[5]:
                imp = 0
                cnt = 0
                for p in convnext_block.parameters():
                    imp+=(p.grad*p).abs().mean()
                    cnt+=1
                imp/=cnt
                block_imp.append(imp)
            block_imp = torch.stack(block_imp)
            _, idx = torch.sort(block_imp)
            keeped_layers = model.features[5][:-args.remove_layers]
            print("Keeped layers: ", keeped_idxs)
            model.features[5] = nn.Sequential(*keeped_layers)

        if isinstance(model, timm.models.convnext.ConvNeXt):
            block_imp = []
            for convnext_block in model.stages[-2].blocks:
                imp = 0
                cnt = 0
                for p in convnext_block.parameters():
                    imp+=(p.grad*p).abs().mean()
                    cnt+=1
                imp/=cnt
                block_imp.append(imp)
            block_imp = torch.stack(block_imp)
            _, idx = torch.sort(block_imp, descending=True)
            # keep the layers with largest importance
            keeped_idxs = idx[:-args.remove_layers]
            keeped_idxs.sort()
            keeped_layers = []
            print("Keeped layers: ", keeped_idxs)
            for i, block in enumerate(model.stages[-2].blocks):
                if i in keeped_idxs:
                    keeped_layers.append(block)
            model.stages[-2].blocks = nn.Sequential(*keeped_layers)
            peval.utils.set_timm_drop_path(model, args.drop_path)

    print("========================================")
    print(model)
    #print("Removed layers: ", idx[-args.remove_layers:])

    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        acc_pruned, loss_pruned = validate_model(model, val_loader, device)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

    print("----------------------------------------")
    print("Summary:")
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("MACs: %.2f G => %.2f G"%(base_macs/1e9, pruned_macs/1e9))
    print("Params: %.2f M => %.2f M"%(base_params/1e6, pruned_params/1e6))
    if args.test_accuracy:
        print("Loss: %.4f => %.4f"%(loss_ori, loss_pruned))
        print("Accuracy: %.4f => %.4f"%(acc_ori, acc_pruned))

    if args.save_as is not None:
        print("Saving the pruned model to %s..."%args.save_as)
        os.makedirs(os.path.dirname(args.save_as), exist_ok=True)
        model.zero_grad()
        torch.save(model, args.save_as)

if __name__=='__main__':
    main()