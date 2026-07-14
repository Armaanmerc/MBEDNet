import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from .modules import (
    ResBlock3D, MPSKPyramidPooling, TSMambaBlock, CBFFM, ECA3D, PEE3D,
)


class MbEdNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, features=None,
                 d_state=16, d_conv=4, expand=2, elrs_rank=16):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.features = features
        self.num_classes = num_classes
        num_enc = len(features) - 1

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, features[0], 7, stride=2, padding=3),
            nn.InstanceNorm3d(features[0]),
            nn.GELU())

        self.cnn_blocks = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        self.cbffm_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(num_enc):
            ch = features[i]
            self.cnn_blocks.append(nn.Sequential(ResBlock3D(ch, ch), MPSKPyramidPooling(ch)))
            self.mamba_blocks.append(
                TSMambaBlock(ch, num_layers=1, d_state=d_state, d_conv=d_conv, expand=expand))
            self.cbffm_blocks.append(CBFFM(ch * 2, ch))
            self.downsamplers.append(nn.Sequential(
                nn.Conv3d(ch, features[i + 1], 3, stride=2, padding=1),
                nn.InstanceNorm3d(features[i + 1]),
                nn.GELU()))

        self.bottleneck = nn.Sequential(*[
            TSMambaBlock(features[-1], num_layers=1, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(2)])

        self.mask_ups = nn.ModuleList()
        self.mask_eca = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.deep_heads = nn.ModuleList()
        for i in range(num_enc - 1, -1, -1):
            up_in = features[i + 1]
            out_ch = features[i]
            self.mask_ups.append(nn.ConvTranspose3d(up_in, out_ch, 2, stride=2))
            self.mask_eca.append(ECA3D(features[i]))
            self.mask_convs.append(nn.Sequential(
                nn.Conv3d(out_ch * 2, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch), nn.GELU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch), nn.GELU()))
            self.deep_heads.append(nn.Conv3d(out_ch, num_classes, 1))

        self.edge_ups = nn.ModuleList()
        self.edge_pees = nn.ModuleList()
        for i in range(num_enc - 1, -1, -1):
            up_in = features[i + 1]
            skip_ch = features[i]
            out_ch = features[i]
            self.edge_ups.append(nn.ConvTranspose3d(up_in, out_ch, 2, stride=2))
            self.edge_pees.append(PEE3D(out_ch + skip_ch, out_ch))
        self.vert_fusions = nn.ModuleList([
            nn.Sequential(nn.Conv3d(features[i] * 2, features[i], 1), nn.GELU())
            for i in range(num_enc - 1, -1, -1)])
        self.edge_head = nn.Conv3d(features[0], 1, 1)

        self.mask_head = nn.Conv3d(features[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        mamba_children = set()
        for m in self.modules():
            if isinstance(m, Mamba):
                for sub in m.modules():
                    mamba_children.add(sub)
        for m in self.modules():
            if m in mamba_children:
                continue
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for cnn, mamba, cbffm, down in zip(
                self.cnn_blocks, self.mamba_blocks, self.cbffm_blocks, self.downsamplers):
            fused = cbffm(cnn(x), mamba(x))
            skips.append(fused)
            x = down(fused)

        x = self.bottleneck(x)
        mask_x = x
        edge_x = x
        deep_outputs = []

        for i in range(len(self.mask_ups)):
            skip = skips[-(i + 1)]
            mask_x = self.mask_ups[i](mask_x)
            skip_att = self.mask_eca[i](skip)
            if skip_att.shape[2:] != mask_x.shape[2:]:
                skip_att = F.interpolate(skip_att, size=mask_x.shape[2:], mode="trilinear", align_corners=False)
            mask_x = self.mask_convs[i](torch.cat([mask_x, skip_att], dim=1))

            edge_x = self.edge_ups[i](edge_x)
            skip_e = skip
            if skip_e.shape[2:] != edge_x.shape[2:]:
                skip_e = F.interpolate(skip_e, size=edge_x.shape[2:], mode="trilinear", align_corners=False)
            edge_x = self.edge_pees[i](torch.cat([edge_x, skip_e], dim=1))
            ex = edge_x
            if ex.shape[2:] != mask_x.shape[2:]:
                ex = F.interpolate(ex, size=mask_x.shape[2:], mode="trilinear", align_corners=False)
            mask_x = self.vert_fusions[i](torch.cat([mask_x, ex], dim=1))

            ds = self.deep_heads[i](mask_x)
            scale = 2 ** (len(self.features) - 1 - i)
            if scale > 1:
                ds = F.interpolate(ds, scale_factor=scale, mode="trilinear", align_corners=False)
            deep_outputs.append(ds)

        mask_logits = self.mask_head(mask_x)
        mask_logits = F.interpolate(mask_logits, scale_factor=2, mode="trilinear", align_corners=False)
        edge_logits = self.edge_head(edge_x)
        edge_logits = F.interpolate(edge_logits, scale_factor=2, mode="trilinear", align_corners=False)
        return mask_logits, edge_logits, deep_outputs
