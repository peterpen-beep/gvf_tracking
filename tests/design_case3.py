"""Case 3 场景设计与扰动灵敏度扫描（用于确认论文对比结论可复现）。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mgvfdc.config import GVFConfig, GuidanceConfig, SimConfig
from mgvfdc.controllers import MGVFDCController, make_controller
from mgvfdc.metrics import summarize
from mgvfdc.scenarios import case3
from mgvfdc.sim import run_simulation

GVF, CTRL = GVFConfig(), GuidanceConfig()
SIM = SimConfig(dt=0.002, duration=60.0)


def run(tag: str, scenario, kind: str) -> dict:
    res = run_simulation(scenario, make_controller(kind, GVF, CTRL), SIM)
    s = summarize(res)
    print(f"  {tag:<24} 最小安全距离 {s['min_clearance']:7.3f} m | "
          f"制导角RMSE {s['rmse_e_psi_far']:.4f} | 稳态RMSE "
          f"{s['ss_rmse_err_x']:.4f}/{s['ss_rmse_err_y']:.4f} | "
          f"末端距目标 {s['final_dist']:7.4f} m | 路径 {s['path_length']:6.2f} m")
    return s


print("=" * 120)
print("一、窄门宽度选择（d_w3 = 0.4，论文给定扰动）")
print("=" * 120)
for off in (1.5, 1.7, 2.0, 2.4):
    print(f"\n门半宽 offset = {off} m（通道宽 ≈ {2*(off-2.2):.1f} m）")
    sc = case3(60.0, gate_offset=off)
    run("MGVFDC(有补偿)", sc, "mgvfdc")
    run("MGVF(无补偿)", sc, "mgvf")

print("\n" + "=" * 120)
print("二、扰动灵敏度扫描（d_w = [0, 0, d3]，gate_offset = 1.7）")
print("=" * 120)
for d3 in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    print(f"\nd_w3 = {d3}")
    sc = case3(60.0, d_w=(-0.1, 0.05, d3))
    run("MGVFDC(有补偿)", sc, "mgvfdc")
    run("MGVF(无补偿)", sc, "mgvf")
    run("GVFPID(基线)", sc, "gvfpid")
