"""对比基线：GVFPID（论文引文 [31] 的方法重建）。

论文第 V 节用 "GVFPID" 代表引文 [31]（M. Lv et al., Control Eng. Pract. 2023）的
"经典 GVF + PID 制导控制"方法，用于证明本文改进（直线化吸引场 + 可调融合函数 +
GPIO 抗扰）的优越性。由于 [31] 的实现细节未在本文复述，本文件按其方法学要点重建：

  1. 向量场：经典 GVF，属性向量固定（att_mode="fixed"）→ 积分曲线为椭圆扇区，
     存在绕远路现象（论文 Fig. 3(a) 的现象）；
  2. 制导：对参考航向角用 PID（而非 GPIO+PD），对纵向速度用距离 PID
     （而非 GPIO+滑模），因此**不具备不确定扰动补偿能力**。

为公平对比，基线与被测方法共享同一套物理约束（u_max / r_max）与同一场景。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import GVFConfig, GuidanceConfig
from .controllers import ControlOut
from .gvf import resultant_field
from .plant import wrap_angle


@dataclass(frozen=True)
class PIDGains:
    """PID 增益（【整定】论文未给基线参数，按常规整定）。"""

    kp_yaw: float = 1.2
    ki_yaw: float = 0.25
    kd_yaw: float = 0.35
    kp_dist: float = 0.55
    ki_dist: float = 0.30


class GVFPIDController:
    """经典 GVF + PID 制导控制器（基线）。"""

    name = "GVFPID"

    def __init__(self,
                 gvf_cfg: GVFConfig | None = None,
                 ctrl_cfg: GuidanceConfig | None = None,
                 gains: PIDGains | None = None):
        base = gvf_cfg or GVFConfig()
        # 强制使用经典（固定属性向量）吸引场，复现论文 Fig. 3(a) 的绕远路现象
        self.gvf = GVFConfig(**{**base.__dict__, "att_mode": "fixed"})
        self.cfg = ctrl_cfg or GuidanceConfig()
        self.gains = gains or PIDGains()

        self.int_yaw = 0.0
        self.int_dist = 0.0
        self.psi_r_prev: float | None = None

    # ------------------------------------------------------------------
    def reset(self, eta_b: np.ndarray, psi: float = 0.0,
              obstacle_pos: Sequence[np.ndarray] = ()) -> None:
        self.int_yaw = 0.0
        self.int_dist = 0.0
        self.psi_r_prev = None

    # ------------------------------------------------------------------
    def step(self,
             eta_b: np.ndarray,
             psi: float,
             psi_dot: float,
             eta_t: np.ndarray,
             obstacle_pos: Sequence[np.ndarray],
             obstacle_rho: Sequence[float],
             dt: float) -> ControlOut:
        cfg, g = self.cfg, self.gains
        eta_e = np.asarray(eta_b, dtype=float) - np.asarray(eta_t, dtype=float)

        _, diag = resultant_field(eta_b, eta_t, obstacle_pos, obstacle_rho,
                                  self.gvf, psi_r_prev=self.psi_r_prev)
        psi_r = diag.psi_r

        # ---- 转向 PID ----
        # 符号约定与式 (22) 一致：e_ψ = ψ − ψ_r > 0（机头偏左）时需给出负的 r（右转）
        e_psi = wrap_angle(psi - psi_r)
        self.int_yaw += e_psi * dt
        self.int_yaw = float(np.clip(self.int_yaw, -5.0, 5.0))      # 抗积分饱和
        r_cmd = -g.kp_yaw * e_psi - g.ki_yaw * self.int_yaw - g.kd_yaw * psi_dot
        r_cmd = float(np.clip(r_cmd, -cfg.r_max, cfg.r_max))

        # ---- 趋近 PID（按到目标的距离，带抗积分饱和 + 期望间距） ----
        d_ref = 0.2                                  # 期望保持的跟随间距 [m]
        dist = float(np.linalg.norm(eta_e))
        self.int_dist += (dist - d_ref) * dt
        self.int_dist = float(np.clip(self.int_dist, -3.0, 3.0))     # 抗积分饱和
        u_raw = g.kp_dist * (dist - d_ref) + g.ki_dist * self.int_dist
        if abs(e_psi) > cfg.psi_T:                                   # 同样受转向优先约束
            gain = float(np.exp(cfg.psi_T - abs(e_psi)))
            u_raw *= gain
        else:
            gain = 1.0
        u_cmd = float(np.clip(u_raw, cfg.u_min, cfg.u_max))

        self.psi_r_prev = psi_r

        return ControlOut(
            u=u_cmd, r=r_cmd, psi_r=psi_r,
            s=float(eta_e[0] + cfg.lam * eta_e[1]), e_psi=e_psi,
            d_w3_hat=0.0, d_p_hat=0.0, u_raw=float(u_raw),
            priority_gain=gain,
            sigma_min=float(np.min(diag.sigmas)) if len(diag.sigmas) else 1.0,
            active=bool(diag.active),
        )
