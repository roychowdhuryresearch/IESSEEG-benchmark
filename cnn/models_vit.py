import torch
import torch.nn as nn
import math

class PatchEmbed3D(nn.Module):
    """
    Splits a 3D volume into (D_patch, H_patch, W_patch) patches and maps each to a vector of size embed_dim.
    Essentially acts like a 'Conv3D' with kernel_size=patch_size and stride=patch_size,
    then flattens the result to get patch tokens.
    """
    def __init__(
        self,
        in_channels=1,
        embed_dim=128,
        patch_size=(2, 4, 4)
    ):
        super().__init__()
        self.patch_size = patch_size
        # A simple linear projection to embed each patch. 
        # Conv3d with kernel_size=patch_size, stride=patch_size => each patch becomes one "token."
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, D, H, W)
        Output shape: (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)  # => (batch_size, embed_dim, D//patchD, H//patchH, W//patchW)
        # Flatten the spatial dimensions
        x = x.flatten(2)  # => (batch_size, embed_dim, num_patches)
        # Transpose to (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block with multi-head self-attention + MLP.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # so input shape is (batch, seq, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class VisionTransformer3D(nn.Module):
    """
    A simplified 3D Vision Transformer:
      - PatchEmbed3D => patchify the 3D volume
      - Learnable class token + positional embeddings
      - Stacked TransformerEncoderBlocks
      - Classification head
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        # 1) Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # Calculate sequence length based on input dimensions and patch size
        # Input shape is (batch, 1, 22, 64, 2000)
        self.seq_len = (22 // patch_size[0]) * (64 // patch_size[1]) * (2000 // patch_size[2])
        
        # 2) CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3) Positional embedding - now sized for our actual sequence length
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim))  # +1 for CLS token
        # A smaller dropout on embeddings
        self.pos_drop = nn.Dropout(p=dropout)

        # 4) Stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 5) Final classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pos_embed, std=1e-6)
        for p in self.head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x => (batch_size, in_channels, D, H, W)

        # 1) Patch embedding => shape (batch, num_patches, embed_dim)
        x = self.patch_embed(x)

        # 2) Prepend CLS token
        #    Expand cls_token to batch size, then cat along patch-dimension
        b, n, _ = x.shape
        cls_token = self.cls_token.expand(b, -1, -1)  # => (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # => (batch_size, 1+num_patches, embed_dim)

        # 3) Add positional embeddings 
        x = x + self.pos_embed  # No need to slice anymore since size matches
        x = self.pos_drop(x)

        # 4) Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) Final norm + classification head using CLS token
        x = self.norm(x)
        cls = x[:, 0]  # cls token is first
        out = self.head(cls)
        return out

def vit3d(
    in_channels=1,
    num_classes=2,
    patch_size=(2, 4, 4),
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.1
):
    """
    Constructor-like function to build a 3D Vision Transformer 
    with some default hyperparameters.
    """
    model = VisionTransformer3D(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )
    return model
