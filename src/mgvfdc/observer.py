"""广义比例积分观测器（GPIO）——论文式 (23)、式 (28)。

论文的 GPIO 结构（以偏航通道为例，式 (23)）：

    ψ̂˙   = r + d̂_11 − ω_11 (ψ̂ − ψ)
    d̂˙_11 = d̂_12 − ω_12 (ψ̂ − ψ)
      ⋮
    d̂˙_1m = −ω_1m (ψ̂ − ψ)
    d̂_w3 = d̂_11

误差动态（式 (24)）为 Ė = A·E + D，其中 A 为伴矩阵：
    A = [[−ω_11, 1, 0, ..., 0],
         [−ω_12, 0, 1, ..., 0],
         ...
         [−ω_1m, 0, 0, ..., 0]]
其特征多项式 s^m + ω_11 s^(m−1) + ... + ω_1m，选参数使其 Hurwitz 即可得
d̂_w3 → d_w3（Theorem 2）。本实现用极点配置 ω_i = C(m,i)·ω_o^i。

内部模型视角：该观测器能无差跟踪"m−1 次多项式型"扰动；对常值扰动取 m≥1 即可。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import GPIOConfig


class GPIO:
    """m 阶 GPIO。

    参数
    ----
    cfg : GPIOConfig
        观测器阶数与带宽（增益由极点配置自动生成）。
    x0 : float
        被观测状态初值（通常取首次测量值）。
    """

    def __init__(self, cfg: GPIOConfig, x0: float = 0.0):
        self.cfg = cfg
        self.w: Tuple[float, ...] = cfg.gains
        self.m = cfg.order
        self.x_hat = float(x0)
        # d_hat[0] 估计扰动本身，d_hat[1] 估计其一阶导数，依此类推
        self.d_hat = np.zeros(self.m, dtype=float)

    # ------------------------------------------------------------------
    def reset(self, x0: float = 0.0) -> None:
        self.x_hat = float(x0)
        self.d_hat = np.zeros(self.m, dtype=float)

    # ------------------------------------------------------------------
    def step(self, x_meas: float, v_known: float, dt: float) -> Tuple[float, float]:
        """推进一个控制步。

        参数
        ----
        x_meas : float
            状态测量值（偏航通道为 ψ，滑模通道为 s）。
        v_known : float
            已知输入（偏航通道为 r；滑模通道为 (cosψ+λsinψ)·u）。
        dt : float
            步长。

        返回
        ----
        (x_hat, d_hat_1) ：状态估计与扰动估计。
        """
        err = self.x_hat - x_meas
        w = self.w

        # 状态方程：ẋ̂ = v + d̂_1 − ω_1·err
        x_dot = v_known + self.d_hat[0] - w[0] * err

        # 扰动及其各阶导数。★注意下标★：第 j 行（j 从 1 数）用的是 ω_j，因此
        #   d̂₁' = d̂₂ − ω₂·err     ← 不是 ω₁！
        #   d̂₂' = d̂₃ − ω₃·err
        #   ...
        #   d̂_m' = −ω_m·err
        # 对应代码里的切片 w[1:]（写成 w[:-1] 是常见错误：那样第一列会变成
        # [ω₁, ω₁, ω₂, …]，特征多项式从 (s+ω_o)^m 退化为 s³+ω₁s²+ω₁s+ω₂，
        # 极点跑到 [−23.3, −0.338±2.85j]，观测器变成弱阻尼振荡器、d̂ 永不收敛）。
        d_dot = np.empty(self.m, dtype=float)
        if self.m > 1:
            d_dot[:-1] = self.d_hat[1:] - np.asarray(w[1:], dtype=float) * err
        d_dot[-1] = -w[-1] * err

        # 显式欧拉（步长足够小；极点均在 −ω_o，欧拉稳定条件 dt·ω_o ≪ 2）
        self.x_hat += dt * x_dot
        self.d_hat += dt * d_dot
        return self.x_hat, float(self.d_hat[0])

    # ------------------------------------------------------------------
    @property
    def disturbance(self) -> float:
        """当前扰动估计 d̂（= d̂_1）。"""
        return float(self.d_hat[0])

    def state(self) -> np.ndarray:
        """完整内部状态 [x̂, d̂_1, ..., d̂_m]，便于外部记录/重置。"""
        return np.concatenate([[self.x_hat], self.d_hat])
