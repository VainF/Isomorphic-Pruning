import os, sys
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import timm
import torch_pruning as tp
import pbench
pbench.forward_patch.patch_timm_forward() # patch timm.forward() to support pruning on ViT


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ResNet Pruning')
    
    # Model
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    parser.add_argument('--ckpt', default=None, type=str, help='model name')
    parser.add_argument('--is-torchvision', default=False, action='store_true', help='use torchvision model')
    # Data
    parser.add_argument('--data-path', default='data/imagenet', type=str, help='model name')
    parser.add_argument('--disable-imagenet-mean-std', default=False, action='store_true', help='use imagenet mean and std')
    parser.add_argument('--train-batch-size', default=64, type=int, help='train batch size')
    parser.add_argument('--val-batch-size', default=64, type=int, help='val batch size')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation mode', choices=['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'])
    parser.add_argument('--val-resize', default=256, type=int, help='resize size')
    
    args = parser.parse_args()
    return args

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=True, interpolation='bicubic', val_resize=256):
    """The imagenet_root should contain train and val folders.
    """
    interpolation = getattr(T.InterpolationMode, interpolation.upper())

    print('Parsing dataset...')
    train_dst = ImageFolder(os.path.join(imagenet_root, 'train'), 
                            transform=pbench.data.presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=val_resize,
                                interpolation=interpolation,
                            )
    )
    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), 
                          transform=pbench.data.presets.ClassificationPresetEval(
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_inputs = torch.randn(1,3,224,224)
    train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, use_imagenet_mean_std = not args.disable_imagenet_mean_std, val_resize=args.val_resize, interpolation=args.interpolation)

    # Load the model
    
    if os.path.isfile(args.model):
        print("Loading model %s..."%args.model)
        state = torch.load(args.model, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state:
                model = state['model']
            elif 'state_dict_ema' in state:
                model = state['state_dict_ema'] # compatible to convnext EMA
            elif 'state_dict' in state:
                model = state['state_dict']
            else:
                raise ValueError("Invalid checkpoint")
        else:
            model = state
        model.eval()
    elif args.is_torchvision:
        import torchvision.models as models
        print("Loading torchvision model %s..."%args.model)
        model = models.__dict__[args.model](pretrained=True).eval()
    else:
        print("Loading timm model %s..."%args.model)
        model = timm.create_model(args.model, pretrained=True).eval()
    if args.ckpt is not None:
        print("Loading checkpoint from %s..."%args.ckpt)
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    model.to(device)
    print(model)
    input_size = [3, 224, 224]
    example_inputs = torch.randn(1, *input_size).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("MACs: %.4f G, Params: %.4f M"%(base_macs/1e9, base_params/1e6))
    print("Evaluating %s..."%(args.model))
    acc_ori, loss_ori = validate_model(model, val_loader, device)
    print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

if __name__=='__main__':
    main()