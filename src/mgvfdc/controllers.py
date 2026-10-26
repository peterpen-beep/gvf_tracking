"""制导控制器：论文第 IV 节完整实现。

  式 (21) 转向误差      ė_ψ = ψ̇ − ψ̇_r = r + d_w3 − ψ̇_r
  式 (22) 转向控制律    r = −k1 e_ψ − k2 ė_ψ − d̂_w3      （PD + GPIO 补偿）
  式 (23) 偏航通道 GPIO
  式 (26) 趋近用模型    ẋ = u cosψ + d_w1, ẏ = u sinψ + d_w2
  式 (27) 滑模控制律    u = (−k3 s − k4 sign(s) − d̂_p) / (cosψ + λ sinψ)
  式 (28) 滑模通道 GPIO
  式 (29)-(31) Theorem 3 的有限时间收敛证明
  式 (32)(33) 转向优先策略

本文件另含三处【实现必备】的工程化处理（论文未提及，见 README）：
  E1. 式 (22) 的 ė_ψ 含 r 本身（代数环）→ 用测量的偏航角速度 ψ̇_meas 代替。
  E2. 式 (27) 的分母 cosψ+λsinψ 在 ψ ≈ arctan(−1/λ) 时过零 → 奇异保护 + u/r 饱和。
  E3. ψ̇_meas 必须是 IMU 式"直接角速度测量"；若用航向差分，−k2·ψ̇ 项会引入一步延迟，
      离散闭环变为 r_k = −k2·r_{k−1} + …，当 k2 ≥ 1 时发散（实测为 ±r_max 高频抖振），
      故 k2 需取在 (0,1) 内。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import GVFConfig, GuidanceConfig
from .gvf import circulation_sense, resultant_field
from .observer import GPIO
from .plant import wrap_angle


@dataclass
class ControlOut:
    """单步控制输出与诊断量。"""

    u: float                 # 纵向速度指令
    r: float                 # 偏航角速度指令
    psi_r: float             # 参考航向角（GVF 制导角）
    s: float                 # 滑模面 s = x_e + λ y_e
    e_psi: float             # 转向误差 ψ − ψ_r
    d_w3_hat: float          # 偏航扰动估计
    d_p_hat: float           # 趋近通道合成扰动估计
    u_raw: float             # 未饱和/未加权的原始纵向指令
    priority_gain: float     # 转向优先缩放系数 exp(ψ_T − |e_ψ|)
    sigma_min: float         # 最小融合权重（诊断是否进入纯斥力区）
    active: bool             # 是否处于避障激活状态


class MGVFDCController:
    """改进 GVF + GPIO 扰动补偿制导控制器（论文完整方法，简称 MGVFDC）。

    compensation=False 时退化为论文中的对照方法 MGVF（无扰动补偿机制）。
    """

    name = "MGVFDC"

    def __init__(self,
                 gvf_cfg: GVFConfig | None = None,
                 ctrl_cfg: GuidanceConfig | None = None):
        self.gvf = gvf_cfg or GVFConfig()
        self.cfg = ctrl_cfg or GuidanceConfig()
        if not self.cfg.compensation:
            self.name = "MGVF"

        self.yaw_gpio = GPIO(self.cfg.yaw_obs)
        self.s_gpio = GPIO(self.cfg.s_obs)

        self.psi_r_prev: float | None = None
        self.psi_prev: float | None = None
        self._senses: list[float] = []
        self._sense_latched: list[float] = []

    # ------------------------------------------------------------------
    def reset(self, eta_b: np.ndarray, psi: float = 0.0,
              obstacle_pos: Sequence[np.ndarray] = ()) -> None:
        self.yaw_gpio.reset(psi)
        self.psi_r_prev = None
        self.psi_prev = psi
        self._senses = []
        self._sense_latched = []

    # ------------------------------------------------------------------
    def _resolve_senses(self, eta_b: np.ndarray, eta_t: np.ndarray,
                        obstacle_pos: Sequence[np.ndarray],
                        sigmas: np.ndarray) -> list[float]:
        """逐障碍物确定绕行方向；进入激活区后锁定该方向（避免旋向抖动）。"""
        senses: list[float] = []
        for i, eta_o in enumerate(obstacle_pos):
            if i >= len(self._sense_latched):
                # 未进入激活区（σ=1）时用当前几何计算候选方向
                s_new = circulation_sense(eta_b, eta_t, eta_o, self.gvf.rep_sense_rule)
                self._sense_latched.append(s_new)
                senses.append(s_new)
            else:
                active = bool(len(sigmas) > i and sigmas[i] < 1.0 - 1e-12)
                if active:
                    senses.append(self._sense_latched[i])      # 锁定
                else:
                    self._sense_latched[i] = circulation_sense(
                        eta_b, eta_t, eta_o, self.gvf.rep_sense_rule)
                    senses.append(self._sense_latched[i])
        return senses

    # ------------------------------------------------------------------
    def step(self,
             eta_b: np.ndarray,
             psi: float,
             psi_dot: float,
             eta_t: np.ndarray,
             obstacle_pos: Sequence[np.ndarray],
             obstacle_rho: Sequence[float],
             dt: float) -> ControlOut:
        """一个控制步。psi_dot 为测量到的偏航角速度。"""
        cfg = self.cfg
        beta, diag_eta = eta_b, eta_t
        eta_e = np.asarray(beta, dtype=float) - np.asarray(diag_eta, dtype=float)

        # ---------------- 1) GVF：制导角 ψ_r（式 (4)-(20)） ----------------
        # 先用上一时刻的 senses 计算 σ 以决定是否锁定（两遍计算，代价可忽略）
        _, diag = resultant_field(eta_b, eta_t, obstacle_pos, obstacle_rho,
                                  self.gvf, senses=self._sense_latched or None,
                                  psi_r_prev=self.psi_r_prev)
        senses = self._resolve_senses(eta_b, eta_t, obstacle_pos, diag.sigmas)
        _, diag = resultant_field(eta_b, eta_t, obstacle_pos, obstacle_rho,
                                  self.gvf, senses=senses, psi_r_prev=self.psi_r_prev)
        psi_r = diag.psi_r
        self._senses = senses

        # ψ̇_r：数值微分 + 限幅（抑制场方向突变处的尖峰）
        if self.psi_r_prev is None:
            psi_r_dot = 0.0
        else:
            psi_r_dot = wrap_angle(psi_r - self.psi_r_prev) / dt
            psi_r_dot = float(np.clip(psi_r_dot, -cfg.psi_r_dot_max, cfg.psi_r_dot_max))

        # ---------------- 2) 转向控制律（式 (21)(22)） ----------------
        e_psi = wrap_angle(psi - psi_r)                      # e_ψ = ψ − ψ_r
        # 【E1】ė_ψ 用测量角速度，避免 r 出现在两端造成代数环
        e_psi_dot = psi_dot - psi_r_dot
        d_w3_hat = self.yaw_gpio.disturbance if cfg.compensation else 0.0
        r_raw = -cfg.k1 * e_psi - cfg.k2 * e_psi_dot - d_w3_hat
        r_cmd = float(np.clip(r_raw, -cfg.r_max, cfg.r_max))

        # 偏航 GPIO（式 (23)）：已知输入为实际施加的 r
        self.yaw_gpio.step(psi, r_cmd, dt)

        # ---------------- 3) 趋近平滑模控制律（式 (26)(27)） ----------------
        s = float(eta_e[0] + cfg.lam * eta_e[1])             # 滑模面 s = x_e + λ y_e
        den = float(np.cos(psi) + cfg.lam * np.sin(psi))
        # 【E2】分母奇异保护
        if abs(den) < cfg.den_guard:
            den = cfg.den_guard if den >= 0.0 else -cfg.den_guard

        d_p_hat = self.s_gpio.disturbance if cfg.compensation else 0.0
        switch = np.tanh(s / cfg.eps_tanh)                   # tanh 替代 sign，抑制抖振
        u_raw = (-cfg.k3 * s - cfg.k4 * switch - d_p_hat) / den

        # ---------------- 4) 转向优先策略（式 (32)(33)） ----------------
        if abs(e_psi) > cfg.psi_T:
            gain = float(np.exp(cfg.psi_T - abs(e_psi)))
            u_raw = gain * u_raw
        else:
            gain = 1.0
        u_cmd = float(np.clip(u_raw, cfg.u_min, cfg.u_max))

        # 滑模通道 GPIO（式 (28)）：已知输入为 (cosψ+λsinψ)·u
        self.s_gpio.step(s, den * u_cmd, dt)

        # ---------------- 5) 状态记录 ----------------
        self.psi_r_prev = psi_r
        self.psi_prev = psi

        return ControlOut(
            u=u_cmd, r=r_cmd, psi_r=psi_r, s=s, e_psi=e_psi,
            d_w3_hat=float(self.yaw_gpio.disturbance),
            d_p_hat=float(self.s_gpio.disturbance),
            u_raw=float(u_raw), priority_gain=gain,
            sigma_min=float(np.min(diag.sigmas)) if len(diag.sigmas) else 1.0,
            active=bool(diag.active),
        )


# ----------------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------------
def make_controller(kind: str,
                    gvf_cfg: GVFConfig | None = None,
                    ctrl_cfg: GuidanceConfig | None = None):
    """按名称创建控制器：mgvfdc（完整方法）/ mgvf（无补偿）/ gvfpid（基线）。"""
    kind = kind.lower()
    if kind in ("mgvfdc", "mgvf"):
        cfg = ctrl_cfg or GuidanceConfig()
        if kind == "mgvfdc":
            cfg = GuidanceConfig(**{**cfg.__dict__, "compensation": True})
        else:
            cfg = GuidanceConfig(**{**cfg.__dict__, "compensation": False})
        return MGVFDCController(gvf_cfg, cfg)
    if kind == "gvfpid":
        from .baselines import GVFPIDController
        return GVFPIDController(gvf_cfg, ctrl_cfg)
    raise KeyError(f"unknown controller '{kind}'")
