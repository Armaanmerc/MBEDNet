import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class GatedSpatialConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU())
        self.conv1x1 = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.Conv3d(channels, channels, 1),
            nn.GELU())
        self.conv_fuse = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels))

    def forward(self, z):
        return self.conv_fuse(self.conv3x3(z) * self.conv1x1(z))


class TriOrientatedMamba(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.mamba_depth = MambaLayer(channels, d_state, d_conv, expand)
        self.mamba_height = MambaLayer(channels, d_state, d_conv, expand)
        self.mamba_width = MambaLayer(channels, d_state, d_conv, expand)
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 3, channels, 1),
            nn.InstanceNorm3d(channels))

    def _scan_axis(self, x, mamba, axis):
        B, C, D, H, W = x.shape
        if axis == 0:
            t = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B * H * W, D, C)
        elif axis == 1:
            t = x.permute(0, 2, 4, 3, 1).contiguous().reshape(B * D * W, H, C)
        else:
            t = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B * D * H, W, C)
        chunk = 4096
        parts = [mamba(t[i:i + chunk]) for i in range(0, t.size(0), chunk)]
        out = torch.cat(parts, dim=0)
        if axis == 0:
            out = out.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        elif axis == 1:
            out = out.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2)
        else:
            out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return out.contiguous()

    def forward(self, x):
        return self.fusion(torch.cat([
            self._scan_axis(x, self.mamba_depth, 0),
            self._scan_axis(x, self.mamba_height, 1),
            self._scan_axis(x, self.mamba_width, 2),
        ], dim=1))


class TSMambaBlock(nn.Module):
    def __init__(self, channels, num_layers=1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gsc": GatedSpatialConv(channels),
                "norm1": nn.LayerNorm(channels),
                "tom": TriOrientatedMamba(channels, d_state, d_conv, expand),
                "norm2": nn.LayerNorm(channels),
                "mlp": nn.Sequential(
                    nn.Linear(channels, channels * 4), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(channels * 4, channels), nn.Dropout(0.1)),
            }) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer["gsc"](x)
            x_r = layer["norm1"](x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            x = x + layer["tom"](x_r)
            x_r = layer["norm2"](x.permute(0, 2, 3, 4, 1))
            x = x + layer["mlp"](x_r).permute(0, 4, 1, 2, 3)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch))
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class MPSKPyramidPooling(nn.Module):
    def __init__(self, channels, pool_sizes=(2, 4, 8), reduction=4):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.InstanceNorm3d(channels), nn.GELU())
        self.conv3x3 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels), nn.GELU())
        self.conv5x5 = nn.Sequential(
            nn.Conv3d(channels, channels, 5, padding=2),
            nn.InstanceNorm3d(channels), nn.GELU())
        self.pool_branches = nn.ModuleList()
        for ps in pool_sizes:
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(ps),
                nn.Conv3d(channels, channels // len(pool_sizes), 1),
                nn.GELU()))
        pool_ch = (channels // len(pool_sizes)) * len(pool_sizes)
        self.pool_fuse = nn.Sequential(
            nn.Conv3d(channels + pool_ch, channels, 1),
            nn.InstanceNorm3d(channels), nn.GELU())
        mid = max(channels // reduction, 8)
        self.sk_pool = nn.AdaptiveAvgPool3d(1)
        self.sk_fc = nn.Sequential(
            nn.Linear(channels, mid), nn.GELU(),
            nn.Linear(mid, channels * 3))

    def forward(self, x):
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        size = x.shape[2:]
        pooled = [F.interpolate(b(x), size=size, mode="trilinear", align_corners=False)
                  for b in self.pool_branches]
        f_pool = self.pool_fuse(torch.cat([x] + pooled, dim=1))
        f_sum = f1 + f3 + f5
        b, c = f_sum.shape[:2]
        gap = self.sk_pool(f_sum).view(b, c)
        attn = self.sk_fc(gap).view(b, 3, c)
        attn = F.softmax(attn, dim=1)
        a1, a3, a5 = attn[:, 0], attn[:, 1], attn[:, 2]
        f_sk = (f1 * a1.view(b, c, 1, 1, 1)
                + f3 * a3.view(b, c, 1, 1, 1)
                + f5 * a5.view(b, c, 1, 1, 1))
        return x + f_sk + f_pool


class CBFFM(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU())
        mid = max(out_ch // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_ch, mid),
            nn.GELU(),
            nn.Linear(mid, out_ch),
            nn.Sigmoid())

    def forward(self, x_cnn, x_mamba):
        x = self.conv(torch.cat([x_cnn, x_mamba], dim=1))
        w = self.se(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ECA3D(nn.Module):
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, 1, c)
        y = torch.sigmoid(self.conv(y)).view(b, c, 1, 1, 1)
        return x * y


class PEE3D(nn.Module):
    def __init__(self, in_ch, out_ch, pool_kernels=(3, 5)):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.InstanceNorm3d(out_ch), nn.GELU())
        self.avg_pools = nn.ModuleList([
            nn.AvgPool3d(kernel_size=k, stride=1, padding=k // 2)
            for k in pool_kernels])
        n_concat = len(pool_kernels) + 1
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch * n_concat, out_ch, 1),
            nn.InstanceNorm3d(out_ch), nn.GELU())

    def forward(self, x):
        f_sq = self.squeeze(x)
        edge_feats = [f_sq - pool(f_sq) for pool in self.avg_pools]
        return self.fuse(torch.cat([f_sq] + edge_feats, dim=1))
