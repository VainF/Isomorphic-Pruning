
import timm
import torch_pruning as tp
import torch.nn as nn

class SwinPatchMergingPruner(tp.BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: list):
        tp.prune_linear_out_channels(layer.reduction, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs) -> nn.Module:
        dim = layer.dim
        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]
        tp.prune_linear_in_channels(layer.reduction, idxs_repeated)
        tp.prune_layernorm_out_channels(layer.norm, idxs_repeated)
        return layer

    def get_out_channels(self, layer):
        return layer.reduction.out_features

    def get_in_channels(self, layer):
        return layer.dim


EXTENDED_PRUNERS = {
    timm.models.swin_transformer.PatchMerging: SwinPatchMergingPruner()
}

