# -*- encoding: utf-8 -*-

from __future__ import annotations
import torch
import torch.nn as nn
from src.loss.aspect_loss import AspectLossFunc as AspectLoss
from src.loss.slope_loss import SlopeLossFunc as SlopeLoss


class Loss(object):
    """init loss module
    """

    def __init__(self, weight) -> None:
        self.weight = weight

    def l1_loss(self, sr: torch.Tensor, hr: torch.Tensor, feats = [], epoch: int = -1) -> torch.Tensor:
        mask = ~torch.isnan(hr) 
        criterion = nn.L1Loss()
        loss_pixel = criterion(sr[mask], hr[mask])
        loss_SAR = 0
        if len(feats) >= 4:
            mliA = feats[0]
            mliD = feats[1]
            ccA = feats[2]
            ccD = feats[3]
            loss_asc = criterion(sr[mask] * ccA[mask], hr[mask] * ccA[mask]) + \
                       criterion(sr[mask] * mliA[mask], hr[mask] * mliA[mask])
            loss_desc = criterion(sr[mask] * ccD[mask], hr[mask] * ccD[mask]) + \
                        criterion(sr[mask] * mliD[mask], hr[mask] * mliD[mask])
            
            loss_SAR = (loss_asc + loss_desc) * 0.5
        temp_hr = torch.where(mask, hr, sr) 
        loss_slope = self.SlopeLoss(sr, temp_hr, resolution=2)
        loss_aspect = self.AspectLoss(sr, temp_hr, resolution=2)
        total_loss = loss_pixel + 0.1 * loss_SAR + 0.5 * loss_slope + 0.1 * loss_aspect 
        return total_loss
    
    def l2_loss(self, x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        criterion = nn.MSELoss(reduction=reduction)
        loss = criterion(x, y)
        return loss

    def SlopeLoss(self, sr, hr, resolution):
        criterion = SlopeLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss
    
    def AspectLoss(self, sr, hr, resolution):
        criterion = AspectLoss(epsilon=1e-8, resolution=resolution)
        loss = criterion(sr, hr)
        return loss