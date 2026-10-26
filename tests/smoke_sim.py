"""闭环仿真冒烟测试：快速跑通 Case 1 / Case 2 / Case 3 并打印指标（用于调试）。"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mgvfdc.config import GVFConfig, GuidanceConfig, SimConfig
from mgvfdc.controllers import MGVFDCController, make_controller
from mgvfdc.metrics import summarize
from mgvfdc.scenarios import case1, case2, case3, fig5_scenario
from mgvfdc.sim import run_simulation


def _line(tag: str, res, extra: str = "") -> None:
    s = summarize(res)
    print(f"  {tag:<26} 末端距目标 {s['final_dist']:7.4f} m | 稳态RMSE x/y "
          f"{s['ss_rmse_err_x']:.4f}/{s['ss_rmse_err_y']:.4f} m | "
          f"制导角RMSE(远) {s['rmse_e_psi_far']:.4f} rad | "
          f"最小安全距离 {s['min_clearance']:6.3f} m | "
          f"σ_min {res.sigma_min.min():.4f} | 路径 {s['path_length']:6.2f} m{extra}")


def main() -> None:
    sim = SimConfig(dt=0.002, duration=120.0)
    gvf, ctrl = GVFConfig(), GuidanceConfig()

    print("=" * 130)
    print("冒烟测试（论文参数 ρ_b=1, ρ_a=2.1, ρ_s=1.1, χ=0.25；安全判据：最小安全距离 > ρ_b = 1.0 m）")
    print("=" * 130)

    for name, scenario in (("Case 1", case1()), ("Case 2", case2()), ("Case 3", case3())):
        print(f"\n--- {name} ---")
        _line("MGVFDC(完整方法)", run_simulation(scenario, MGVFDCController(gvf, ctrl), sim))
        _line("MGVF(无扰动补偿)", run_simulation(scenario, make_controller("mgvf", gvf, ctrl), sim))
        _line("GVFPID(基线)", run_simulation(scenario, make_controller("gvfpid", gvf, ctrl), sim))
        no_avoid = GVFConfig(**{**gvf.__dict__, "rep_strength": 0.0})
        _line("参考: 关闭避障(纯追踪)", run_simulation(scenario, MGVFDCController(no_avoid, ctrl), sim))

    print("\n--- Fig5 场景：χ 扫描（论文 Fig.5：χ 增大 ⇒ 与障碍边界距离缩短） ---")
    for chi in (0.1, 0.25, 0.5, 0.75, 0.9):
        r = run_simulation(fig5_scenario(45.0),
                           MGVFDCController(GVFConfig(chi=chi), ctrl),
                           SimConfig(dt=0.002, duration=45.0))
        print(f"  χ={chi:<5} 最小安全距离 = {r.safety_margin():.3f} m, "
              f"最终距目标 = {r.dist[-1]:.5f} m")


if __name__ == "__main__":
    main()
