"""全局配置：论文给定参数 + 需要自行整定的参数（含推导依据）。

参数来源标注：
  【论文】= 论文第 V 节明确给出的数值
  【整定】= 论文未给出，由本实现按理论推导或工程惯例确定（见 README 复现缺口清单）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Tuple


# ----------------------------------------------------------------------------
# GPIO 观测器配置（论文式 (23)(28)）
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GPIOConfig:
    """m 阶广义比例积分观测器（Generalized Proportional Integral Observer）。

    误差动态（论文式 (24)）的特征多项式为
        s^m + w1*s^(m-1) + w2*s^(m-2) + ... + wm
    【整定】把全部观测器极点配置在 -omega_o，即 (s+omega_o)^m，
    则增益为 wi = C(m, i) * omega_o^i，i = 1..m。
    """

    order: int = 3                 # 【整定】GPIO 阶数 m（论文未给；m=3 可跟踪 2 次多项式型扰动）
    omega_o: float = 8.0           # 【整定】观测器带宽 [rad/s]（取制导环带宽的 3~5 倍）

    @property
    def gains(self) -> Tuple[float, ...]:
        """返回 (w1, ..., wm)，对应极点全在 -omega_o 的二项式系数配置。"""
        m, wo = self.order, self.omega_o
        return tuple(comb(m, j) * (wo ** j) for j in range(1, m + 1))


# ----------------------------------------------------------------------------
# 制导向量场（GVF）配置（论文式 (4)-(20)）
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GVFConfig:
    """改进 GVF 向量场参数。"""

    rho_b: float = 1.0                          # 【论文】机器人安全圆半径 [m]
    rho_a: float = 2.1                          # 【论文】避障激活距离（障碍边界外）[m]
    rho_s: float = 1.1                          # 【论文】最小安全距离（障碍边界外）[m]
    chi: float = 0.25                           # 【论文】融合函数可调参数 χ ∈ (0,1)

    att_mode: str = "variable"                  # variable→式(8)改进场；fixed→经典 GVF
    fixed_p: Tuple[float, float] = (1.0, 0.0)   # 【整定】经典 GVF 的固定属性向量（缺口#1）
    rep_sense_rule: str = "side"                # 【整定】绕行方向规则：side 保持同侧 / progress 朝目标
    rep_strength: float = 1.0                   # 【整定】斥力场强度系数（缺口#1 的标度自由度）
    normalize_fields: bool = True               # 【整定】是否把两个场单位化后再融合（推荐，见 README）

    # ---- 合成场 (12) 中"吸引项权重"的读法（勘误记录）----
    # 论文式 (12) 原文为：  F = ∏_i σ_i(β_i) · F_a(η_e) + Σ_i (1−σ_i) · F_r,i
    #     —— 吸引项是**连乘 ∏**（障碍物只削弱目标吸引力，绝不产生吸引）。
    #     连乘的物理含义：任意一个障碍物使 σ_i = 0，整个吸引项立刻归零，
    #     机器人只剩斥力兜底 ⇒ 结构上杜绝吸引力把机器人拖进障碍物。
    #
    # "product" = ∏_i σ_i   ← 论文原文，默认（本实现按此执行）
    # "min"     = min_i σ_i ← 等价稳健写法（同样满足"任一为 0 则整体为 0"）
    # "sum"     = Σ_i σ_i   ← 【勘误】本实现早期误把 ∏ 读成 Σ 的写法。
    #                          多障碍时会放大吸引项（因 PDF 提取器丢掉了连乘号字形），
    #                          实测 Case 1 会径直穿入障碍（安全距离 −0.38 m）。
    #                          仅保留用于消融对照，不要用于正式实验。
    blend: str = "product"
    result_eps: float = 1e-9                    # 合成场幅值过小时的保护阈值


# ----------------------------------------------------------------------------
# 制导控制器配置（论文式 (21)-(33)）
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GuidanceConfig:
    """GPIO-PD 转向 + GPIO-SMC 趋近 + 转向优先。"""

    k1: float = 2.0                # 【整定】转向 PD 比例项（闭环时间常数 ≈ (1+k2)/k1）
    k2: float = 0.6                # 【整定】转向 PD 微分项；⚠ 必须 < 1，见 README 实现问题 E3
    k3: float = 1.0                # 【整定】滑模线性增益（决定 s 的指数收敛速率）
    k4: float = 0.5                # 【整定】滑模切换增益（需满足 Theorem 3：k4 > Δ）
    lam: float = 1.0               # 【整定】滑模面 s = x_e + λ y_e 的权重 λ

    eps_tanh: float = 0.01         # 【整定】tanh 去抖宽度（论文建议用 tanh 替代 sign）
    psi_T: float = 0.45            # 【整定】转向优先阈值 ψ_T [rad]（缺口#3，论文只说是"经验值"）

    # u 的取值域说明（见 README 实现问题 E4）：
    # 论文式 (27) 的分母 (cosψ+λsinψ) 与滑模面 s 在"机头对准目标"的正常工况下符号相反，
    # 因此 u 自动为正；但若机头尚未转向目标（例如固定 p 的经典 GVF 把参考航向指向背离目标
    # 的一侧），分子分母同号会要求 u < 0。四足机器人 Remark 1 只禁止横荡输入 v ≡ 0，
    # 并未允许倒车，故默认令 u ∈ [0, u_max]（"转向优先"策略与此哲学一致：先转向再前进）。
    # 把 u_min 设为 −u_max 即得到"有符号纵向速度"变体，可复现固定 p 场的闭环绕行轨迹。
    u_max: float = 1.2             # 【整定】纵向速度上限 [m/s]（四足机器人物理约束）
    u_min: float = 0.0             # 纵向速度下限（0 = 仅前进，见上方说明）
    r_max: float = 1.5             # 【整定】偏航角速度上限 [rad/s]
    den_guard: float = 0.12        # 【整定】式(27)分母 cosψ+λsinψ 的奇异保护阈值（缺口#6）

    compensation: bool = True      # True→MGVFDC（含扰动补偿）；False→MGVF（无补偿）
    psi_r_dot_max: float = 6.0     # 【整定】参考航向角速度限幅 [rad/s]（抑制数值微分尖峰）

    yaw_obs: GPIOConfig = field(default_factory=lambda: GPIOConfig(order=3, omega_o=8.0))
    s_obs: GPIOConfig = field(default_factory=lambda: GPIOConfig(order=3, omega_o=6.0))


# ----------------------------------------------------------------------------
# 仿真配置
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """闭环仿真参数。"""

    dt: float = 0.002              # 控制/积分步长 [s]（500 Hz）
    duration: float = 120.0        # 单次仿真时长 [s]
    d_w: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # 集总扰动 d_w = [d_w1, d_w2, d_w3]
    meas_noise_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # 位置/航向测量噪声（0=理想）
    yaw_rate_noise: float = 0.0    # 角速度测量噪声（IMU 传感器噪声，0=理想）


# 【论文】Case 3 的常值扰动
PAPER_DISTURBANCE = (-0.1, 0.05, 0.4)


def paper_gvf_config(**overrides) -> GVFConfig:
    """论文第 V 节参数下的默认 GVF 配置。"""
    return GVFConfig(**overrides)


def paper_guidance_config(**overrides) -> GuidanceConfig:
    """论文默认制导控制器配置。"""
    return GuidanceConfig(**overrides)
