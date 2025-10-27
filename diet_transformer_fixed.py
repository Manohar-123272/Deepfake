import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DietTransformerBlock(nn.Module):
    """
    Diet Transformer Block - Lightweight transformer with shared parameters
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Shared QKV projection to reduce parameters
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        # Lightweight MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Multi-head self-attention
        shortcut = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class PatchEmbedding(nn.Module):
    """
    Patch Embedding for images
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class DietTransformerDeepfake(nn.Module):
    """
    Diet Transformer for Deepfake Detection
    Lightweight transformer with parameter sharing and efficient attention
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks with parameter sharing (Diet concept)
        # Use fewer unique blocks and share parameters
        self.shared_blocks = nn.ModuleList([
            DietTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth // 2)  # Reduced number of unique blocks
        ])
        
        # Layer sharing pattern
        self.layer_sharing = [0, 1, 0, 1, 0, 1][:depth]  # Repeat pattern for diet approach
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # For GradCAM compatibility
        self.transformer = nn.ModuleDict({
            'layers': self.shared_blocks
        })
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        
        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks with parameter sharing
        for layer_idx in self.layer_sharing:
            if layer_idx < len(self.shared_blocks):
                x = self.shared_blocks[layer_idx](x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use cls token)
        x = self.head(x[:, 0])
        
        return x

class DietTransformerAdvanced(nn.Module):
    """
    Advanced Diet Transformer with more sophisticated parameter sharing
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Shared attention and MLP components
        self.shared_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True)
            for _ in range(3)  # 3 shared attention modules
        ])
        
        self.shared_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(drop_rate)
            ) for _ in range(3)  # 3 shared MLP modules
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(6)  # 6 layer norms
        ])
        
        # Define sharing pattern
        self.attention_pattern = [0, 1, 2, 0, 1, 2] * (depth // 6 + 1)
        self.mlp_pattern = [0, 1, 2, 0, 1, 2] * (depth // 6 + 1)
        self.depth = depth
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # For GradCAM compatibility
        self.transformer = nn.ModuleDict({
            'layers': self.shared_attention
        })
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        
        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer layers with parameter sharing
        for i in range(self.depth):
            # Attention with residual connection
            attn_idx = self.attention_pattern[i % len(self.attention_pattern)]
            mlp_idx = self.mlp_pattern[i % len(self.mlp_pattern)]
            
            # Pre-norm
            x_norm = self.layer_norms[i % 6](x)
            
            # Self-attention
            attn_out, _ = self.shared_attention[attn_idx](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # MLP with residual connection
            x_norm = self.layer_norms[(i + 3) % 6](x)
            mlp_out = self.shared_mlp[mlp_idx](x_norm)
            x = x + mlp_out
        
        # Final norm
        x = self.final_norm(x)
        
        # Classification
        x = self.head(x[:, 0])
        
        return x

# Helper function to create different Diet Transformer variants
def create_diet_transformer(variant='base', **kwargs):
    """
    Create different variants of Diet Transformer
    """
    if variant == 'tiny':
        return DietTransformerDeepfake(
            embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, **kwargs)
    elif variant == 'small':
        return DietTransformerDeepfake(
            embed_dim=256, depth=8, num_heads=4, mlp_ratio=4, **kwargs)
    elif variant == 'base':
        return DietTransformerDeepfake(
            embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    elif variant == 'advanced':
        return DietTransformerAdvanced(
            embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")