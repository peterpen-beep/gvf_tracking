"""诊断脚本：逐时间步打印闭环内部量，定位跟踪失效原因。"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mgvfdc.config import GVFConfig, GuidanceConfig, SimConfig
from mgvfdc.controllers import MGVFDCController
from mgvfdc.gvf import resultant_field
from mgvfdc.plant import Unicycle, obstacle_arrays, wrap_angle
from mgvfdc.scenarios import case1


def main() -> None:
    scenario = case1(20.0)
    dt = 0.002
    steps = int(20.0 / dt)
    gvf, ctrl = GVFConfig(), GuidanceConfig()
    robot = Unicycle(*scenario.start, scenario.psi0)
    controller = MGVFDCController(gvf, ctrl)
    obs0, rho0 = obstacle_arrays(scenario.obstacles, 0.0)
    controller.reset(robot.eta, robot.psi, obs0)

    # 静态场检查
    F, diag = resultant_field(robot.eta, scenario.target.position(0.0),
                              obs0, rho0, gvf)
    print(f"初始：F = {F}, ψ_r = {diag.psi_r:.4f}, sigmas = {np.round(diag.sigmas, 3)}")
    print(f"      目标方向 = {np.arctan2(scenario.target.position(0.0)[1]-robot.y, scenario.target.position(0.0)[0]-robot.x):.4f}")
    print(f"      初始 ψ = {robot.psi:.4f}")

    print("\n    t      x       y     tgt_x   tgt_y  |e|     psi    psi_r   e_psi     u      r      s     den   sig_min  a_pri")
    psi_prev = robot.psi
    for k in range(steps):
        t = k * dt
        eta_t = scenario.target.position(t)
        obs_pos, obs_rho = obstacle_arrays(scenario.obstacles, t)
        psi_dot = wrap_angle(robot.psi - psi_prev) / dt
        psi_prev = robot.psi
        out = controller.step(robot.eta, robot.psi, psi_dot, eta_t, obs_pos, obs_rho, dt)
        if k % int(0.5 / dt) == 0:
            den = np.cos(robot.psi) + ctrl.lam * np.sin(robot.psi)
            d = float(np.linalg.norm(robot.eta - eta_t))
            print(f"{t:6.2f} {robot.x:7.2f} {robot.y:7.2f} {eta_t[0]:7.2f} {eta_t[1]:7.2f} "
                  f"{d:6.2f} {robot.psi:7.3f} {out.psi_r:7.3f} {out.e_psi:7.3f} "
                  f"{out.u:6.3f} {out.r:6.3f} {out.s:8.3f} {den:6.3f} {out.sigma_min:6.3f} "
                  f"{out.priority_gain:5.2f}")
        robot.step(out.u, out.r, scenario.d_w, dt)


if __name__ == "__main__":
    main()
