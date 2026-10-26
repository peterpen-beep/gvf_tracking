"""闭环仿真引擎：把制导控制器接到式 (1) 的独轮车模型上。

数据流：
    GVF(位置) → ψ_r → 转向控制律 r → 转向优先 → 速度指令 u → 式(1) 积分 → 新位姿
                     ↑ GPIO(d̂_w3)                    ↑ GPIO(d̂_p)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .config import SimConfig
from .controllers import ControlOut
from .plant import Unicycle, obstacle_arrays, safety_clearance, wrap_angle
from .scenarios import Scenario


@dataclass
class SimResult:
    """一次仿真的完整日志（时间序列 + 汇总）。"""

    name: str
    controller: str
    scenario: str
    dt: float
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    psi_r: np.ndarray
    s: np.ndarray
    e_psi: np.ndarray
    d_w3_hat: np.ndarray
    d_p_hat: np.ndarray
    u_raw: np.ndarray
    priority_gain: np.ndarray
    sigma_min: np.ndarray
    tgt_x: np.ndarray
    tgt_y: np.ndarray
    err_x: np.ndarray
    err_y: np.ndarray
    dist: np.ndarray
    clearance: np.ndarray            # (n_obs, T)：到各障碍物边界的距离
    obstacle_xy: np.ndarray          # (n_obs, 2, T)：障碍物轨迹（画图用）
    obstacle_rho: np.ndarray         # (n_obs,)
    meta: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_obs(self) -> int:
        return int(self.clearance.shape[0])

    def min_clearance(self) -> np.ndarray:
        """每个障碍物在整个过程中的最小安全距离指标。"""
        if self.clearance.size == 0:
            return np.zeros(0)
        return self.clearance.min(axis=1)

    def safety_margin(self) -> float:
        """全局最小安全距离指标（与论文 ρ_b 比较）。"""
        if self.clearance.size == 0:
            return float("inf")
        return float(self.clearance.min())


def run_simulation(scenario: Scenario,
                   controller,
                   sim_cfg: SimConfig | None = None,
                   record_obstacles: bool = True) -> SimResult:
    """运行一次闭环仿真。"""
    sim_cfg = sim_cfg or SimConfig()
    dt = sim_cfg.dt
    steps = int(round(scenario.duration / dt)) + 1
    rng = np.random.default_rng(0)

    robot = Unicycle(scenario.start[0], scenario.start[1], scenario.psi0)

    # 初始时刻的障碍物位置，用于初始化控制器内部的绕行方向与观测器
    obs0, rho0 = obstacle_arrays(scenario.obstacles, 0.0)
    controller.reset(robot.eta, robot.psi, obs0)

    n_obs = len(scenario.obstacles)

    # 预分配日志
    log = {k: np.zeros(steps) for k in
           ("t", "x", "y", "psi", "u", "r", "psi_r", "s", "e_psi",
            "d_w3_hat", "d_p_hat", "u_raw", "priority_gain", "sigma_min",
            "tgt_x", "tgt_y", "err_x", "err_y", "dist")}
    clearance = np.zeros((n_obs, steps))
    obstacle_xy = np.zeros((n_obs, 2, steps))
    noise = np.asarray(sim_cfg.meas_noise_std, dtype=float)

    psi_meas_prev = robot.psi
    r_applied = 0.0
    out: ControlOut | None = None

    for k in range(steps):
        t = k * dt
        eta_t = scenario.target.position(t)
        obs_pos, obs_rho = obstacle_arrays(scenario.obstacles, t)

        # ---- 测量（可加噪声；默认理想） ----
        eta_meas = robot.eta.copy()
        psi_meas = robot.psi
        if np.any(noise > 0):
            eta_meas = eta_meas + rng.normal(0.0, noise[0], size=2)
            psi_meas = wrap_angle(psi_meas + rng.normal(0.0, noise[1]))

        # 偏航角速度测量：IMU 直接测量机体角速度 ω = r + d_w3，而不是对航向求差分。
        # 【实现问题 E3】若用航向差分 ψ̇ ≈ (ψ_k − ψ_{k−1})/dt，则测得的角速度滞后一步等于
        # 上一步的 r，控制律中的 −k2·ψ̇ 会形成 r_k = −k2·r_{k−1} + … 的离散递推，
        # 当 |k2| ≥ 1 时发散（表现为 r 在 ±r_max 之间高频抖振）。真实 IMU 无此延迟。
        psi_dot_meas = r_applied + scenario.d_w[2]
        if sim_cfg.yaw_rate_noise > 0.0:
            psi_dot_meas += float(rng.normal(0.0, sim_cfg.yaw_rate_noise))
        psi_meas_prev = psi_meas

        # ---- 制导控制 ----
        out = controller.step(eta_meas, psi_meas, psi_dot_meas, eta_t,
                              obs_pos, obs_rho, dt)

        # ---- 记录（用真实位姿评估性能） ----
        eta = robot.eta
        clr = safety_clearance(eta, scenario.obstacles, t)
        log["t"][k] = t
        log["x"][k], log["y"][k], log["psi"][k] = robot.x, robot.y, robot.psi
        log["u"][k], log["r"][k] = out.u, out.r
        log["psi_r"][k] = out.psi_r
        log["s"][k] = out.s
        log["e_psi"][k] = out.e_psi
        log["d_w3_hat"][k] = out.d_w3_hat
        log["d_p_hat"][k] = out.d_p_hat
        log["u_raw"][k] = out.u_raw
        log["priority_gain"][k] = out.priority_gain
        log["sigma_min"][k] = out.sigma_min
        log["tgt_x"][k], log["tgt_y"][k] = eta_t[0], eta_t[1]
        log["err_x"][k] = eta[0] - eta_t[0]
        log["err_y"][k] = eta[1] - eta_t[1]
        log["dist"][k] = float(np.linalg.norm(eta - eta_t))
        if n_obs:
            clearance[:, k] = clr
            for i, p in enumerate(obs_pos):
                obstacle_xy[i, :, k] = p

        # ---- 被控对象积分（式 (1)） ----
        robot.step(out.u, out.r, scenario.d_w, dt)
        r_applied = out.r

    meta = {
        "min_safety_margin": float(clearance.min()) if n_obs else float("inf"),
        "collided": float(clearance.min() < 1.0) if n_obs else 0.0,
        "final_dist": float(log["dist"][-1]),
    }

    return SimResult(
        name=scenario.name,
        controller=getattr(controller, "name", controller.__class__.__name__),
        scenario=scenario.name,
        dt=dt,
        obstacle_rho=np.asarray(rho0, dtype=float),
        obstacle_xy=obstacle_xy,
        clearance=clearance,
        meta=meta,
        **log,
    )


# ----------------------------------------------------------------------------
# 批量对比
# ----------------------------------------------------------------------------
def run_comparison(scenario: Scenario,
                   controllers: Dict[str, object],
                   sim_cfg: SimConfig | None = None) -> Dict[str, SimResult]:
    """对同一场景运行多个控制器。"""
    results: Dict[str, SimResult] = {}
    for key, ctrl in controllers.items():
        results[key] = run_simulation(scenario, ctrl, sim_cfg)
    return results
