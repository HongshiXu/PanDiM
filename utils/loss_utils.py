from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict

class HybridL1SSIM(torch.nn.Module):
    def __init__(self, channel=31, weighted_r=(1.0, 0.1)):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self._l1 = torch.nn.L1Loss()
        self._ssim = SSIMLoss(channel=channel)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss


class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class LossWarpper(torch.nn.Module):
    def __init__(self, weighted_ratio=(1.0, 1.0), **losses):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(losses.keys())
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt):
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            loss += l
            d_loss[n] = l
        return loss, d_loss


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()

class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 防止除零错误

    def forward(self, pred, target):
        # pred: 生成图像 (B, C, H, W)
        # target: 参考图像 (B, C, H, W)
        # 计算光谱角度
        dot_product = (pred * target).sum(dim=1)  # (B, H, W)
        pred_norm = torch.norm(pred, p=2, dim=1)  # (B, H, W)
        target_norm = torch.norm(target, p=2, dim=1)  # (B, H, W)
        cosine_similarity = dot_product / (pred_norm * target_norm + self.eps)  # (B, H, W)
        sam = torch.acos(torch.clamp(cosine_similarity, -1 + self.eps, 1 - self.eps))  # (B, H, W)
        return sam.mean()  # 返回平均SAM值
    
class HybridL1SAM(nn.Module):
    def __init__(self, lambda_sam=1.0, lambda_l1=1.0, eps=1e-8):
        super().__init__()
        self.lambda_sam = lambda_sam  # SAM损失权重
        self.lambda_l1 = lambda_l1  # L1损失权重
        # self.lambda_spatial = lambda_spatial  # 空间细节损失权重
        self.sam_loss = SAMLoss(eps=eps)
        self.l1_loss = nn.L1Loss()

    # def spatial_loss(self, pred, target):
    #     # 计算梯度差异作为空间细节损失
    #     pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    #     pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    #     target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    #     target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    #     loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    #     loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    #     return (loss_x + loss_y) / 2

    def forward(self, pred, target):
        # pred: 生成图像 (B, C, H, W)
        # target: 参考图像 (B, C, H, W)
        sam_loss = self.sam_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        # spatial_loss = self.spatial_loss(pred, target)
        # total_loss = self.lambda_sam * sam_loss + self.lambda_l1 * l1_loss + self.lambda_spatial * spatial_loss
        total_loss = self.lambda_sam * sam_loss + self.lambda_l1 * l1_loss
        return total_loss


class SpatialFrequencyFilter(nn.Module):
    """
    频域滤波基类：在空间域利用可微高斯核分离高/低频，绝对锁定像素相位。
    GT_FDDL 和 Spatial_FDDL 共享此基类。
    """
    def __init__(self, kernel_size=5, sigma=1.5):
        super().__init__()
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)
        self.pad = kernel_size // 2

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """生成 2D 高斯核"""
        grid_x, grid_y = torch.meshgrid(
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            indexing='ij'
        )
        center = kernel_size // 2
        dist_sq = (grid_x - center)**2 + (grid_y - center)**2
        kernel = torch.exp(-dist_sq / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def get_low_freq(self, x):
        """低通滤波，提取图像低频成分"""
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B * C, 1, H, W)
        x_padded = F.pad(x_reshaped, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        low_freq = F.conv2d(x_padded, self.gaussian_kernel)
        return low_freq.reshape(B, C, H, W)

    def get_high_freq(self, x):
        """高通滤波，提取图像高频纹理和边缘"""
        return x - self.get_low_freq(x)


class GT_FDDL(SpatialFrequencyFilter):
    """
    GT-Referenced Frequency Decoupled Loss (面向 Ground Truth 的高频强调损失)
    
    直接对 GT 进行频率分解作为优化目标，不使用代理信号 (LRMS/PAN)，
    因此不引入系统偏差。通过 high_boost > 1 强化高频约束，迫使网络
    学习 GT 的锐利边缘和纹理细节，主攻 PSNR/SSIM。
    """
    def __init__(self, high_boost=2.0, low_weight=1.0, kernel_size=5, sigma=1.5):
        super().__init__(kernel_size, sigma)
        self.high_boost = high_boost
        self.low_weight = low_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, gt):
        """
        Args:
            pred: 模型预测 (B, C, H, W)，可以是残差或完整图像
            gt:   Ground Truth (B, C, H, W)
        """
        loss_low = self.l1_loss(self.get_low_freq(pred), self.get_low_freq(gt))
        loss_high = self.l1_loss(self.get_high_freq(pred), self.get_high_freq(gt))
        return self.low_weight * loss_low + self.high_boost * loss_high


class Spatial_FDDL(SpatialFrequencyFilter):
    """
    Spatial Frequency Decoupled Loss (面向源图像的物理先验损失)
    
    利用跨模态物理先验：低频对齐 LRMS（光谱保真），高频对齐 PAN（结构注入）。
    作为辅助正则项使用，配合余弦衰减在训练后期逐步退出，避免代理偏差累积。
    """
    def __init__(self, weight=1.0, kernel_size=5, sigma=1.5):
        super().__init__(kernel_size, sigma)
        self.weight = weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_hrms, lrms_up, pan):
        """
        Args:
            pred_hrms: 预测的高分辨率多光谱图 (B, C, H, W)
            lrms_up:   上采样的低分辨率多光谱图 (B, C, H, W)
            pan:       高分辨率全色图 (B, 1, H, W)
        """
        # 低频逼近 LRMS (光谱保真)
        loss_low = self.l1_loss(self.get_low_freq(pred_hrms), self.get_low_freq(lrms_up))

        # 高频逼近 PAN (结构注入)
        pred_high = self.get_high_freq(pred_hrms)
        pan_high = self.get_high_freq(pan)
        pred_high_gray = torch.mean(pred_high, dim=1, keepdim=True)
        loss_high = self.l1_loss(pred_high_gray, pan_high)

        return self.weight * (loss_low + loss_high)


# 向后兼容别名
FrequencyDecoupledLoss = Spatial_FDDL



def get_loss(loss_type):
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "l1ssim":
        criterion = HybridL1SSIM(channel=8, weighted_r=(1.0, 0.1))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(channel=31, weighted_r=(1.0, 1.0))
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


if __name__ == "__main__":
    loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    x = torch.randn(1, 31, 64, 64, requires_grad=True)
    y = x + torch.randn(1, 31, 64, 64) / 10
    l = loss(x, y)
    l.backward()
    print(l)
    print(x.grad)

