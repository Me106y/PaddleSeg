# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Restoration-oriented metrics for segmentation outputs.

These complement the region-overlap metrics (mIoU / Acc / Kappa / Dice) that
PaddleSeg reports by default. They are computed on the **binarised foreground
mask** (``pred > 0`` vs ``label > 0``) -- which is what a portrait / whiteboard
restoration pipeline actually consumes downstream:

* ``psnr``  -- peak signal-to-noise ratio between predicted and GT mask.
* ``ssim``  -- structural similarity, Gaussian window of 11x11 / sigma 1.5
               with the standard constants (K1=0.01, K2=0.03, data_range=1).
* ``TemporalConsistency`` -- frame-to-frame stability of the prediction on the
  pixels whose ground truth did *not* change. It is only meaningful when the
  evaluation list is ordered as consecutive video frames; PaddleSeg evaluates
  with ``shuffle=False`` and ``drop_last=False``, so dataset order is kept.

Typical usage::

    tc = TemporalConsistency()
    tc.update(pred, label)          # one sample
    tc.flicker                      # mean |dPred| on unchanged pixels
    tc.score                        # 1 - flicker
"""

import numpy as np
import paddle
import paddle.nn.functional as F

_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_C1 = (_SSIM_K1 * 1.0)**2
_SSIM_C2 = (_SSIM_K2 * 1.0)**2
_DATA_RANGE = 1.0
_PSNR_CAP = 100.0


def _gaussian_kernel(window, sigma):
    coords = np.arange(window) - (window - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g).astype(np.float32)


_KERNEL = _gaussian_kernel(_SSIM_WINDOW, _SSIM_SIGMA)


def foreground_mask(tensor):
    """
    Convert a prediction / label tensor into a float32 foreground indicator.

    Args:
        tensor (Tensor|np.ndarray): prediction or label, shape (N, 1, H, W),
            (N, H, W) or (H, W). Values > 0 are treated as foreground.

    Returns:
        np.ndarray: float32 array of shape (H, W) with values in {0, 1}.
    """
    if isinstance(tensor, paddle.Tensor):
        tensor = tensor.numpy()
    arr = np.asarray(tensor)
    arr = np.squeeze(arr)
    return (arr > 0).astype(np.float32)


def psnr(pred, label):
    """
    PSNR between the binarised prediction and label masks, in dB.
    Returns a capped value (default 100.0) for identical masks.
    """
    p = foreground_mask(pred)
    g = foreground_mask(label)
    mse = float(np.mean((p - g)**2))
    if mse <= 0:
        return _PSNR_CAP
    value = 10.0 * np.log10((_DATA_RANGE**2) / mse)
    return float(min(value, _PSNR_CAP))


def ssim(pred, label):
    """
    SSIM between the binarised prediction and label masks, in [-1, 1].
    """
    p = foreground_mask(pred)
    g = foreground_mask(label)
    x = paddle.to_tensor(p).reshape([1, 1, p.shape[0], p.shape[1]])
    y = paddle.to_tensor(g).reshape([1, 1, g.shape[0], g.shape[1]])
    kernel = paddle.to_tensor(_KERNEL).reshape(
        [1, 1, _SSIM_WINDOW, _SSIM_WINDOW])
    pad = _SSIM_WINDOW // 2

    def filt(v):
        return F.conv2d(v, kernel, padding=pad)

    mu_x, mu_y = filt(x), filt(y)
    sigma_x = filt(x * x) - mu_x * mu_x
    sigma_y = filt(y * y) - mu_y * mu_y
    sigma_xy = filt(x * y) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)
    denominator = (mu_x * mu_x + mu_y * mu_y + _SSIM_C1) * (
        sigma_x + sigma_y + _SSIM_C2)
    return float(paddle.mean(numerator / denominator).numpy())


class RSAccumulator:
    """
    Accumulates PSNR / SSIM over samples and temporal consistency over
    consecutive sample pairs.
    """

    def __init__(self):
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count = 0
        self._flicker_sum = 0.0
        self._flicker_pixels = 0
        self._pair_count = 0
        self._changed_ratio_sum = 0.0
        self._prev_fg = None
        self._prev_gt = None

    def update(self, pred, label):
        cur_fg = foreground_mask(pred)
        cur_gt = foreground_mask(label)
        self._psnr_sum += psnr(pred, label)
        self._ssim_sum += ssim(pred, label)
        self._count += 1

        if self._prev_fg is not None and cur_fg.shape == self._prev_fg.shape:
            same = (cur_gt == self._prev_gt)
            total = cur_gt.size
            if total > 0:
                self._changed_ratio_sum += float(same.size - same.sum()
                                                 ) / total
            n_same = int(same.sum())
            if n_same > 0:
                diff = np.abs(cur_fg - self._prev_fg)[same]
                self._flicker_sum += float(diff.sum())
                self._flicker_pixels += n_same
            self._pair_count += 1

        self._prev_fg = cur_fg
        self._prev_gt = cur_gt

    @property
    def psnr(self):
        return self._psnr_sum / self._count if self._count else 0.0

    @property
    def ssim(self):
        return self._ssim_sum / self._count if self._count else 0.0

    @property
    def flicker(self):
        """Mean |dPred| on GT-unchanged pixels (lower is better)."""
        return self._flicker_sum / self._flicker_pixels \
            if self._flicker_pixels else 0.0

    @property
    def temporal_consistency(self):
        """1 - flicker (higher is better)."""
        return 1.0 - self.flicker

    @property
    def changed_pixel_ratio(self):
        """Mean fraction of GT pixels that changed between frame pairs."""
        return self._changed_ratio_sum / self._pair_count \
            if self._pair_count else 0.0

    def summary(self):
        return dict(
            psnr=self.psnr,
            ssim=self.ssim,
            temporal_consistency=self.temporal_consistency,
            flicker=self.flicker,
            changed_pixel_ratio=self.changed_pixel_ratio,
            samples=self._count,
            consecutive_pairs=self._pair_count)
