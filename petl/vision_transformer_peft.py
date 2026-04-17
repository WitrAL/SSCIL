import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed

import logging

class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a@w1b)*(w2a@w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        w1a, w1b, w2a, w2b, scale = ctx.saved_tensors
        grad_out = grad_out * scale
        
        # 计算W1的梯度
        temp = grad_out*(w2a@w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp
        
        # 计算W2的梯度
        temp = grad_out * (w1a@w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp
        
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None

def make_weight(w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(w1a, w1b, w2a, w2b, scale)

def make_kron(w1, w2, scale):
    """实现Kronecker积"""
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale

def factorization(dimension: int, factor: int=-1) -> tuple[int, int]:
    """将维度分解为两个因子"""
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n

class PEFTLayer:
    def __init__(self, r: int, alpha: int, dropout: float):
        self.r = r
        self.alpha = alpha
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        self.scaling = self.alpha / self.r


    
class LoRALinear(nn.Linear, PEFTLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 r: int = 0, alpha: int = 1, dropout: float = 0.):  # 统一参数名
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        PEFTLayer.__init__(self, r=r, alpha=alpha, dropout=dropout)
        
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # Freeze the original weight
            self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            result = super().forward(x)
            return result + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return super().forward(x)

class LoHALinear(nn.Linear, PEFTLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 r: int = 0, alpha: int = 1, dropout: float = 0.):  # 统一参数名
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        PEFTLayer.__init__(self, r=r, alpha=alpha, dropout=dropout)
        
        if r > 0:
            self.lora_w1_a = nn.Parameter(torch.empty(out_features, r))
            self.lora_w1_b = nn.Parameter(torch.empty(r, in_features))
            self.lora_w2_a = nn.Parameter(torch.empty(out_features, r))
            self.lora_w2_b = nn.Parameter(torch.empty(r, in_features))
            
            # 初始化权重
            nn.init.kaiming_uniform_(self.lora_w1_b, a=math.sqrt(5))
            nn.init.zeros_(self.lora_w1_a)
            nn.init.kaiming_uniform_(self.lora_w2_b, a=math.sqrt(5))
            nn.init.zeros_(self.lora_w2_a)

            # nn.init.normal_(self.lora_w1_b, std=1.0)
            # nn.init.normal_(self.lora_w1_a, std=0.1)
            # nn.init.normal_(self.lora_w2_b, std=1.0)
            # nn.init.normal_(self.lora_w2_a, std=0.1)
            
            self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            result = super().forward(x)
            diff_weight = make_weight(
                self.lora_w1_a, self.lora_w1_b,
                self.lora_w2_a, self.lora_w2_b,
                torch.tensor(self.scaling)
            )
            return result + (self.dropout(x) @ diff_weight.T)
        return super().forward(x)

class LoKRLinear(nn.Linear, PEFTLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 r: int = 0, alpha: int = 1, dropout: float = 0., factor: int = -1):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        PEFTLayer.__init__(self, r=r, alpha=alpha, dropout=dropout)
        
        if r > 0:
            # 分解输入和输出维度
            in_m, in_n = factorization(in_features, factor)
            out_l, out_k = factorization(out_features, factor)
            
            # 较小部分的权重
            self.use_w1 = r >= max(out_l, in_m) / 2
            if self.use_w1:
                self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))
            else:
                self.lokr_w1_a = nn.Parameter(torch.empty(out_l, r))
                self.lokr_w1_b = nn.Parameter(torch.empty(r, in_m))
            
            # 较大部分的权重
            self.use_w2 = r >= max(out_k, in_n) / 2
            if self.use_w2:
                self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
            else:
                self.lokr_w2_a = nn.Parameter(torch.empty(out_k, r))
                self.lokr_w2_b = nn.Parameter(torch.empty(r, in_n))
            
            # 初始化权重
            if self.use_w1:
                nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
            
            if self.use_w2:
                nn.init.zeros_(self.lokr_w2)
            else:
                nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
                nn.init.zeros_(self.lokr_w2_b)
            
            self.weight.requires_grad = False

    def get_weight(self):
        w1 = self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b
        return make_kron(w1, w2, torch.tensor(self.scaling))

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            result = super().forward(x)
            diff_weight = self.get_weight()
            return result + (self.dropout(x) @ diff_weight.T)
        return super().forward(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 r=4, alpha=16, peft_type='lora'):  # 添加peft_type参数
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        LinearClass = {
            'lora': LoRALinear,
            'loha': LoHALinear,
            'lokr': LoKRLinear
        }[peft_type]
        # 使用选定的Linear类创建投影层
        self.q_proj = LinearClass(dim, dim, bias=qkv_bias, r=r, alpha=alpha)
        self.k_proj = LinearClass(dim, dim, bias=qkv_bias, r=r, alpha=alpha)
        self.v_proj = LinearClass(dim, dim, bias=qkv_bias, r=r, alpha=alpha)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearClass(dim, dim, r=r, alpha=alpha)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)
        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 r=4, alpha=16, peft_type='lora'):  # 修改参数名和添加peft_type
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop, 
                            r=r, alpha=alpha, peft_type=peft_type)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 根据peft_type选择不同的实现
        LinearClass = {
            'lora': LoRALinear,
            'loha': LoHALinear,
            'lokr': LoKRLinear
        }[peft_type]

        self.fc1 = LinearClass(dim, mlp_hidden_dim, r=r, alpha=alpha)
        self.act = act_layer()
        self.fc2 = LinearClass(mlp_hidden_dim, dim, r=r, alpha=alpha)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = residual + self.drop_path(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 representation_size=None, distilled=False, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, 
                 norm_layer=None, act_layer=None, weight_init='', 
                 r=4, alpha=16, peft_type='lora'):  # 修改参数名和添加peft_type
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                r=r, alpha=alpha,peft_type=peft_type  # 添加peft_type参数
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

def vit_base_patch16_224_peft(pretrained=False, peft_type='lora',**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        peft_type=peft_type, **kwargs)
    
    #if pretrained:
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # Load the pretrained weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded pretrained weights with message: {msg}')
    
    # Freeze all parameters except LoRA parameters
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_base_patch16_224_in21k_peft(pretrained=False, peft_type='lora', **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),peft_type=peft_type, **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight
    # if pretrained:
    #     checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    #     state_dict = checkpoint_model.state_dict()
    
    # Load the pretrained weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded pretrained weights with message: {msg}')

    # Freeze all parameters except LoRA parameters
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 

    return model