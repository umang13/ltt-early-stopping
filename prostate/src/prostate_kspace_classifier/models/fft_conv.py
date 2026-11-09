"""FFT-based complex convolution blocks (KSPACE-NET style)."""

from __future__ import annotations

from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fft import fftn, ifftn


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Grouped complex matrix multiplication helper."""

    scalar = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])
    real = scalar(a.real, b.real) - scalar(a.imag, b.imag)
    imag = scalar(a.imag, b.real) + scalar(a.real, b.imag)
    out = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    out.real, out.imag = real, imag
    return out.view(out.size(0), -1, *out.shape[3:])


def fft_conv(signal: Tensor, kernel: Tensor, bias: Tensor | None = None, groups: int = 1) -> Tensor:
    """Perform convolution via FFT multiplication."""

    kernel_padding: list[int] = []
    for dim in reversed(range(2, signal.ndim)):
        kernel_padding.extend([0, signal.size(dim) - kernel.size(dim)])
    padded_kernel = F.pad(kernel, kernel_padding)
    kernel_fr = fftn(padded_kernel, dim=tuple(range(2, signal.ndim)))
    kernel_fr.imag.mul_(-1)
    output_fr = complex_matmul(signal, kernel_fr, groups=groups)
    output = ifftn(output_fr, dim=tuple(range(2, signal.ndim)))
    if bias is not None:
        shape = (1, -1) + (signal.ndim - 2) * (1,)
        output = output + bias.view(shape)
    return output


class _FFTConv(nn.Module):
    """Base class for FFT convolution layers supporting complex weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size, dtype=torch.complex64)
        weight = weight / (out_channels * in_channels * kernel_size[0] * kernel_size[1])
        self.weight = nn.Parameter(torch.view_as_real(weight))
        if bias:
            bias_init = torch.zeros(out_channels, dtype=torch.complex64)
            self.bias = nn.Parameter(torch.view_as_real(bias_init))
        else:
            self.bias = None
        self.groups = groups

    def forward(self, signal: Tensor) -> Tensor:
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias) if self.bias is not None else None
        return fft_conv(signal, weight, bias=bias, groups=self.groups)


class FFTConv2d(_FFTConv):
    """2D FFT convolution wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
        )
