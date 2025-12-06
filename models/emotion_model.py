import torch
import torch.nn as nn
import timm

def add_lora_to_linear(linear_layer, r=8, alpha=16):
    """
    Applies Low-Rank Adaptation (LoRA) to a linear layer.
    Adapted from Cell [11] in the notebook.
    """
    in_dim, out_dim = linear_layer.in_features, linear_layer.out_features
    
    linear_layer.lora_A = nn.Linear(in_dim, r, bias=False)
    linear_layer.lora_B = nn.Linear(r, out_dim, bias=False)
    
    nn.init.zeros_(linear_layer.lora_B.weight)
    linear_layer.lora_scale = alpha / r

    for p in linear_layer.parameters():
        p.requires_grad_(False)

    def new_forward(x, orig_forward=linear_layer.forward):
        return orig_forward(x) + linear_layer.lora_scale * linear_layer.lora_B(linear_layer.lora_A(x))
    
    linear_layer.forward = new_forward

def get_model(device='cpu', num_classes=7, model_name='vit_base_patch16_224', lora_r=4):
    """
    Rebuilds the exact ViT + LoRA architecture used for training.
    """
    print(f"Creating {model_name} with LoRA (rank={lora_r})...")
    
    vit = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    for name, module in list(vit.named_modules()):
        if isinstance(module, nn.Linear) and 'attn.qkv' in name:
            add_lora_to_linear(module, r=lora_r)
            
    vit.head = nn.Sequential(
        nn.LayerNorm(vit.head.in_features),
        nn.Dropout(p=0.5),
        vit.head
    )
    
    vit.to(device)
    return vit
