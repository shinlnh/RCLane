"""
RCLane -- model architecture ported to PyTorch (NETWORK part only).

Faithfully ported from the original MindSpore implementation:
    https://github.com/lpplbiubiubiub/RCLane
    (src/rclane/segformer.py, rclane_head.py, layers.py)

This module contains only the network forward pass: image -> 5 dense maps
    seg_map, up_arrow, down_arrow, up_bound, down_bound
It does NOT include: GT label encoding (lane_codec.py), relay-chain decoding,
loss, or the training loop.

Notes on deviations from the original MindSpore code (read before training):
  1. The original MiT ADDS absolute positional embeddings (sizes hardcoded for a
     320x800 input). Standard SegFormer has none. Controlled by `use_pos_embed`:
       - True  -> matches the original, but locks the input to `img_size` and
                  prevents loading ImageNet-pretrained MiT weights.
       - False -> standard SegFormer, allows loading pretrained MiT-b0/b1/b2
                  (recommended for training).
  2. Stage-2 attention in the MindSpore code uses sr_ratios[0] (almost certainly a
     copy-paste bug). Here we use sr_ratios[i] per stage, as in standard SegFormer.
  3. `embedding_dim` (decoder dim) and `middle_dim` are not pinned in the original
     repo (train.py never builds the model). Defaults follow the SegFormer
     convention: embedding_dim = 256 for b0, 768 for b1/b2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  layers.py  (DoubleConv, OutConv, DropPath)
# --------------------------------------------------------------------------- #
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per-sample mask
    mask = x.new_empty(shape).bernoulli_(keep)
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DoubleConv(nn.Module):
    """(conv3x3 -> BN -> ReLU) * 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
#  segformer.py  (Mix Vision Transformer encoder)
# --------------------------------------------------------------------------- #
class DWConv(nn.Module):
    """Depth-wise 3x3 conv used inside the Mix-FFN."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAttention(nn.Module):
    """Attention with spatial-reduction (SR), following SegFormer."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0,
                 proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0,
                 attn_drop=0.0, drop_path_rate=0.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """MiT encoder (SegFormer). Produces 4 feature maps at strides 4/8/16/32."""

    def __init__(self, img_size=(320, 800), in_chans=3,
                 embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4), qkv_bias=True, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1),
                 use_pos_embed=True):
        super().__init__()
        self.depths = depths
        self.use_pos_embed = use_pos_embed
        H, W = img_size

        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        if use_pos_embed:
            # sizes depend on img_size (original hardcodes 320x800 -> 16000/4000/1000/250)
            hw = [(math.ceil(H / s), math.ceil(W / s)) for s in (4, 8, 16, 32)]
            self.pos_embed1 = nn.Parameter(torch.zeros(1, hw[0][0] * hw[0][1], embed_dims[0]))
            self.pos_embed2 = nn.Parameter(torch.zeros(1, hw[1][0] * hw[1][1], embed_dims[1]))
            self.pos_embed3 = nn.Parameter(torch.zeros(1, hw[2][0] * hw[2][1], embed_dims[2]))
            self.pos_embed4 = nn.Parameter(torch.zeros(1, hw[3][0] * hw[3][1], embed_dims[3]))
            for pe in (self.pos_embed1, self.pos_embed2, self.pos_embed3, self.pos_embed4):
                nn.init.trunc_normal_(pe, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias,
                                           drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[0])
                                     for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias,
                                           drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[1])
                                     for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias,
                                           drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[2])
                                     for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias,
                                           drop_rate, attn_drop_rate, dpr[cur + i], sr_ratios[3])
                                     for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def _stage(self, x, patch_embed, blocks, norm, pos_embed):
        x, H, W = patch_embed(x)
        if pos_embed is not None:
            x = x + pos_embed
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        pe = [self.pos_embed1, self.pos_embed2, self.pos_embed3, self.pos_embed4] \
            if self.use_pos_embed else [None, None, None, None]
        c1 = self._stage(x, self.patch_embed1, self.block1, self.norm1, pe[0])
        c2 = self._stage(c1, self.patch_embed2, self.block2, self.norm2, pe[1])
        c3 = self._stage(c2, self.patch_embed3, self.block3, self.norm3, pe[2])
        c4 = self._stage(c3, self.patch_embed4, self.block4, self.norm4, pe[3])
        return [c1, c2, c3, c4]


_MIT_CFG = {
    "b0": dict(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2)),
    "b1": dict(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2)),
    "b2": dict(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3)),
}


def build_segformer_backbone(vision="b1", img_size=(320, 800), use_pos_embed=True):
    cfg = _MIT_CFG[vision]
    return MixVisionTransformer(
        img_size=img_size,
        embed_dims=cfg["embed_dims"],
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        depths=cfg["depths"],
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        use_pos_embed=use_pos_embed,
    )


# --------------------------------------------------------------------------- #
#  rclane_head.py  (MLP-fuse decoder + 5 output branches)
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Linear embedding: (B,C,H,W) -> (B, H*W, embed_dim)."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        return self.proj(x)               # (B, HW, embed_dim)


class RCLaneHead(nn.Module):
    """
    SegFormer-style decoder + 5 conv branches. Each branch outputs 2 channels,
    upsampled back to the input resolution.
    fuse_size = stride-4 feature size (H/4, W/4). out_size = input (H, W).
    """

    def __init__(self, in_channels=(64, 128, 320, 512), embedding_dim=256,
                 middle_dim=64, fuse_size=(80, 200), out_size=(320, 800)):
        super().__init__()
        self.fuse_size = fuse_size
        self.out_size = out_size
        c1, c2, c3, c4 = in_channels

        self.linear_c1 = MLP(c1, embedding_dim)
        self.linear_c2 = MLP(c2, embedding_dim)
        self.linear_c3 = MLP(c3, embedding_dim)
        self.linear_c4 = MLP(c4, embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, middle_dim, 1, bias=False)

        def branch():  # DoubleConv -> OutConv(2)
            return nn.Sequential(DoubleConv(middle_dim, middle_dim), OutConv(middle_dim, 2))

        self.out_seg = branch()     # seg_map        (background / foreground)
        self.up_arrow = branch()    # backward transfer  (dx, dy)
        self.down_arrow = branch()  # forward transfer   (dx, dy)
        self.up_bound = branch()    # backward distance
        self.down_bound = branch()  # forward distance

    def _to_map(self, mlp_out, ref):
        """(B,HW,embed) -> (B,embed,h,w) -> resize to fuse_size."""
        B, _, h, w = ref.shape
        x = mlp_out.permute(0, 2, 1).reshape(B, -1, h, w)
        return F.interpolate(x, size=self.fuse_size, mode="bilinear", align_corners=False)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        _c4 = self._to_map(self.linear_c4(c4), c4)
        _c3 = self._to_map(self.linear_c3(c3), c3)
        _c2 = self._to_map(self.linear_c2(c2), c2)
        _c1 = self._to_map(self.linear_c1(c1), c1)

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(x)
        x = self.dropout(x)

        def up(t):
            return F.interpolate(t, size=self.out_size, mode="bilinear", align_corners=False)

        return {
            "seg_map": up(self.out_seg(x)),
            "up_arrow": up(self.up_arrow(x)),
            "down_arrow": up(self.down_arrow(x)),
            "up_bound": up(self.up_bound(x)),
            "down_bound": up(self.down_bound(x)),
        }


# --------------------------------------------------------------------------- #
#  rclane.py  (wrapper)
# --------------------------------------------------------------------------- #
class RCLane(nn.Module):
    def __init__(self, vision="b1", embedding_dim=None, middle_dim=128,
                 img_size=(320, 800), use_pos_embed=True):
        super().__init__()
        # SegFormer convention: decoder dim = 256 for B0, 768 for B1+
        if embedding_dim is None:
            embedding_dim = 256 if vision == "b0" else 768
        self.backbone = build_segformer_backbone(vision, img_size, use_pos_embed)
        in_ch = _MIT_CFG[vision]["embed_dims"]
        fuse_size = (img_size[0] // 4, img_size[1] // 4)
        self.head = RCLaneHead(in_ch, embedding_dim, middle_dim,
                               fuse_size=fuse_size, out_size=img_size)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)   # dict of 5 maps, each (B, 2, H, W)


# --------------------------------------------------------------------------- #
#  smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for vision in ["b0", "b1", "b2"]:
        model = RCLane(vision=vision, img_size=(320, 800), use_pos_embed=True)
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.randn(2, 3, 320, 800)
        model.eval()
        with torch.no_grad():
            out = model(x)
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        print(f"[{vision}] params={n_param:.1f}M")
        for k, s in shapes.items():
            print(f"        {k:12s} {s}")
        # expected: every map == (2, 2, 320, 800)
        assert all(s == (2, 2, 320, 800) for s in shapes.values()), "wrong output shape!"
    print("OK -- forward runs, all 5 maps are (B, 2, 320, 800).")
