# -*- encoding: utf-8 -*-

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.common import * 

class DTRAN(nn.Module):
    def __init__(self, scale: int = 6, n_feats: int = 32, out_channels: int = 1, mean: float = 0., std: float = 0.):
        super(DTRAN, self).__init__()
        
        self.scale = scale

        self.asc_branch = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.desc_branch = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        
        self.terrain_branch = nn.Sequential(
            nn.Conv2d(3, n_feats, 1),
            nn.InstanceNorm2d(n_feats), 
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )

        self.diff_attention = DiffAttention(n_feats)
        self.terrain_sft = TerrainSFT(n_feats)
        self.body = TRAB(n_feats, n_feats, num_modules=4)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, out_channels, 3, padding=1)
        )

    def forward(self, lr_dem, feats: list) -> torch.Tensor:
        
        base_dem = F.interpolate(lr_dem, size=(192, 192), mode='bilinear', align_corners=False)
        mliA, mliD, ccA, ccD, slope, aspect = feats
        f_asc = self.asc_branch(torch.cat([mliA, ccA, base_dem], dim=1))
        f_desc = self.desc_branch(torch.cat([mliD, ccD, base_dem], dim=1))
        f_sar = self.diff_attention(f_asc, f_desc)
        sar_edge = torch.abs(feats[0] + feats[1]) * 0.5 
        sar_edge = F.interpolate(sar_edge, size=(192, 192)) 
        f_terrain = self.terrain_branch(torch.cat([slope, aspect, sar_edge], dim=1))
        f_guided = self.terrain_sft(f_sar, f_terrain)
        res_feat = self.body(f_guided)
        terrain_residual = self.tail(res_feat + f_guided)
        output = base_dem + terrain_residual
        
        return output