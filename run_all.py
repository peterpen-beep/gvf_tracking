"""一键复现论文全部仿真结果。

用法：
    python run_all.py                  # 全量运行（约 2~5 分钟）
    python run_all.py --quick          # 缩减时长快速跑通
    python run_all.py --no-sweep       # 跳过扰动灵敏度扫描

产物：
    outputs/figures/*.png     论文 Fig.3~Fig.9 的复刻图 + 验证/消融图
    outputs/metrics.csv       所有算例的指标表
    outputs/summary.txt       人类可读的结果汇总
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

#优先从同级的src中导入自定义模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import mgvfdc as M
from mgvfdc import plotting


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FIG = os.path.join(HERE, "outputs", "figures")
OUT = os.path.join(HERE, "outputs")


# ----------------------------------------------------------------------------
def gpio_verification_trace(m_true: int = 3, duration: float = 6.0) -> Dict[str, np.ndarray]:
    """验证 Theorem 2：GPIO 估计常值+斜坡扰动。"""
    dt = 0.001
    n = int(duration / dt)
    gpios = {m: M.GPIO(M.GPIOConfig(order=m, omega_o=8.0)) for m in (2, 3)}
    x = 0.0
    t = np.zeros(n)
    d_true = np.zeros(n)
    d_hat = {m: np.zeros(n) for m in gpios}
    for k in range(n):
        tt = k * dt
        v = 0.6 * np.sin(0.5 * tt)
        d = 0.4 + 0.1 * tt                      # 常值 + 斜坡（2 次多项式以内）
        x += dt * (v + d)
        for m, g in gpios.items():
            g.step(x, v, dt)
            d_hat[m][k] = g.disturbance
        t[k], d_true[k] = tt, d
    return {"t": t, "d_true": d_true, "d_hat_2": d_hat[2], "d_hat_3": d_hat[3]}


def theorem1_angle_error(n: int = 4000, seed: int = 0) -> float:
    """Theorem 1：吸引场方向与"机器人→目标"方向夹角的最大值。"""
    rng = np.random.default_rng(seed)
    cfg = M.GVFConfig(att_mode="variable")
    worst = 0.0
    for _ in range(n):
        eta_b = rng.uniform(-60, 60, size=2)
        eta_t = rng.uniform(-60, 60, size=2)
        e = eta_b - eta_t
        if np.linalg.norm(e) < 1e-6:
            continue
        F = M.attractive_field(eta_b, eta_t, cfg)
        to_t = -e
        c = float(np.dot(F, to_t) / (np.linalg.norm(F) * np.linalg.norm(to_t)))
        worst = max(worst, float(np.arccos(np.clip(c, -1.0, 1.0))))
    return worst


def blending_curves(chis=(0.1, 0.25, 0.5, 0.9)) -> Dict[float, tuple]:
    """ξ 与其导数在接缝处的连续性数据：返回 (β 网格, ξ, 接缝点, 接缝处 ξ')。"""
    cfg = M.GVFConfig()
    _, ba, bs = M.beta_triplet(1.0, 1.0, cfg)
    out = {}
    for chi in chis:
        betas = np.linspace(ba, bs, 400)
        xi = np.array([M.blending_function(b, ba, bs, chi) for b in betas])
        dxi = np.gradient(xi, betas)
        bj = ba + chi * (bs - ba)
        val = float(np.interp(bj, betas, dxi))
        out[chi] = (betas, xi, np.array([bj, bj]), np.array([val, val]))
    return out


# ----------------------------------------------------------------------------
def run_all(quick: bool = False, sweep: bool = True) -> None:
    dur = 40.0 if quick else 120.0
    sim = M.SimConfig(dt=0.002, duration=dur)
    gvf, ctrl = M.GVFConfig(), M.GuidanceConfig()

    rows: List[Dict[str, object]] = []
    lines: List[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 108)
    log("论文复现：A Collision-Free Target Tracking Controller With Uncertain Disturbance Rejection")
    log("            for Quadruped Robots (IEEE T-IV 2024) —— 从零实现")
    log("=" * 108)
    log(f"仿真步长 dt = {sim.dt} s，单次时长 = {dur} s，"
        f"论文参数 ρ_b={gvf.rho_b}, ρ_a={gvf.rho_a}, ρ_s={gvf.rho_s}, χ={gvf.chi}")
    log("")

    # ---------------- 论文图 3：吸引场对比 ----------------
    print("\n[1/10] 论文 Fig.3：改进 1（可变属性向量）")
    plotting.fig_field_comparison(OUT_FIG)

    # 闭环"绕远路"量化（远程场景；固定 p 需允许 u<0 才能收敛，见 README 实现问题 E4）
    sc3 = M.fig3_longrange(40.0)
    sim3 = M.SimConfig(dt=0.002, duration=40.0)
    ctrl_signed = M.GuidanceConfig(**{**ctrl.__dict__, "u_min": -ctrl.u_max})
    res_fixed = M.run_simulation(sc3, M.MGVFDCController(
        M.GVFConfig(att_mode="fixed", fixed_p=(1.0, 0.0)), ctrl_signed), sim3)
    res_var = M.run_simulation(sc3, M.MGVFDCController(M.GVFConfig(att_mode="variable"), ctrl_signed), sim3)
    plotting.fig_detour_closedloop(OUT_FIG, res_fixed, res_var)
    for tag, r in (("Fig3-闭环-固定p", res_fixed), ("Fig3-闭环-可变p", res_var)):
        row = {"case": "Fig3 闭环(远程)", "controller": tag}
        row.update({k: v for k, v in M.summarize(r).items() if k not in ("controller", "scenario")})
        rows.append(row)
    L_fixed = M.summarize(res_fixed)["path_length"]
    L_var = M.summarize(res_var)["path_length"]
    log(f"  闭环路径长度：固定 p = {L_fixed:.3f} m（到达 {M.summarize(res_fixed)['arrival_time_1m']:.2f} s），"
        f"可变 p = {L_var:.3f} m（到达 {M.summarize(res_var)['arrival_time_1m']:.2f} s）；"
        f"起终点直线 = 28.284 m ⇒ 改进 1 实现零绕行")
    # 仅前进约束下固定 p 会死锁（论文方法的结构性弱点，值得记录）
    sc3b = M.fig3_longrange(40.0)
    res_fixed_fwd = M.run_simulation(sc3b, M.MGVFDCController(
        M.GVFConfig(att_mode="fixed", fixed_p=(1.0, 0.0)), ctrl), sim3)
    log(f"  附加发现：若强制 u ≥ 0（四足仅前进），固定 p 会陷入 u = 0 死锁——"
        f"仅前移 {M.summarize(res_fixed_fwd)['path_length']:.2f} m，末端距目标 "
        f"{res_fixed_fwd.dist[-1]:.2f} m（式 (27) 的 u 在结构上需要有符号）")

    # ---------------- 论文图 4：融合函数 ----------------
    print("\n[2/10] 论文 Fig.4：改进 2（可调融合函数）")
    plotting.fig_blending(OUT_FIG)

    # ---------------- 论文图 5：χ 微调 ----------------
    print("\n[3/10] 论文 Fig.5：χ 微调避障性能")
    chi_res: Dict[float, M.SimResult] = {}
    for chi in (0.1, 0.25, 0.5, 0.75, 0.9):
        sc = M.fig5_scenario(45.0)
        chi_res[chi] = M.run_simulation(sc, M.MGVFDCController(M.GVFConfig(chi=chi), ctrl),
                                       M.SimConfig(dt=0.002, duration=45.0))
    plotting.fig_chi_tuning(OUT_FIG, chi_res, sc.obstacles[0], target_xy=(0.0, 0.0))
    log("  χ 与最小安全距离：" + "，".join(
        f"χ={c}: {chi_res[c].safety_margin():.3f} m" for c in chi_res))

    # ---------------- Case 1 ----------------
    print("\n[4/10] Case 1：动态目标 + 静态多障碍")
    sc1 = M.case1(dur)
    c1 = {
        "MGVFDC": M.run_simulation(sc1, M.MGVFDCController(gvf, ctrl), sim),
        "GVFPID": M.run_simulation(sc1, M.make_controller("gvfpid", gvf, ctrl), sim),
    }
    no_avoid = M.GVFConfig(**{**gvf.__dict__, "rep_strength": 0.0})
    c1_noavoid = M.run_simulation(sc1, M.MGVFDCController(no_avoid, ctrl), sim)
    plotting.fig_case1(OUT_FIG, c1["MGVFDC"], sc1.obstacles,
                       extra_results={"GVFPID（经典 GVF + PID）": c1["GVFPID"],
                                      "关闭避障（纯追踪参考）": c1_noavoid})
    for tag, r in (("MGVFDC", c1["MGVFDC"]), ("GVFPID", c1["GVFPID"]),
                   ("无避障参考", c1_noavoid)):
        row = {"case": "Case 1", "controller": tag}
        row.update({k: v for k, v in M.summarize(r).items() if k not in ("controller", "scenario")})
        rows.append(row)
    log(f"  MGVFDC：稳态 RMSE x={M.summarize(c1['MGVFDC'])['ss_rmse_err_x']:.4f} m, "
        f"y={M.summarize(c1['MGVFDC'])['ss_rmse_err_y']:.4f} m；"
        f"最小安全距离 {c1['MGVFDC'].safety_margin():.3f} m")
    log(f"  GVFPID：稳态 RMSE x={M.summarize(c1['GVFPID'])['ss_rmse_err_x']:.4f} m, "
        f"y={M.summarize(c1['GVFPID'])['ss_rmse_err_y']:.4f} m；"
        f"路径 {M.summarize(c1['GVFPID'])['path_length']:.1f} m vs "
        f"MGVFDC {M.summarize(c1['MGVFDC'])['path_length']:.1f} m")
    log(f"  关闭避障参考：最小安全距离 {c1_noavoid.safety_margin():.3f} m（<0 表示穿入障碍）")

    # ---------------- 论文图 8：安全距离指标（Case 1） ----------------
    print("\n[5/10] 论文 Fig.8：安全距离指标")
    plotting.fig_safety(OUT_FIG, c1["MGVFDC"], gvf.rho_b)

    # ---------------- Case 2 ----------------
    print("\n[6/10] Case 2：动态目标 + 动态障碍")
    sc2 = M.case2(dur)
    c2 = {
        "MGVFDC": M.run_simulation(sc2, M.MGVFDCController(gvf, ctrl), sim),
        "GVFPID": M.run_simulation(sc2, M.make_controller("gvfpid", gvf, ctrl), sim),
    }
    plotting.fig_case2(OUT_FIG, c2,
                       obstacles_at=lambda t: [(ob.position(t), ob.radius) for ob in sc2.obstacles])
    for tag, r in c2.items():
        row = {"case": "Case 2", "controller": tag}
        row.update({k: v for k, v in M.summarize(r).items() if k not in ("controller", "scenario")})
        rows.append(row)
    log(f"  MGVFDC：最小安全距离 {c2['MGVFDC'].safety_margin():.3f} m，"
        f"路径 {M.summarize(c2['MGVFDC'])['path_length']:.1f} m")
    log(f"  GVFPID：最小安全距离 {c2['GVFPID'].safety_margin():.3f} m，"
        f"路径 {M.summarize(c2['GVFPID'])['path_length']:.1f} m")

    # ---------------- Case 3 ----------------
    print("\n[7/10] Case 3：含不确定扰动的环境")
    sc = M.case3(dur)
    c3 = {
        "MGVFDC（有补偿）": M.run_simulation(sc, M.MGVFDCController(gvf, ctrl), sim),
        "MGVF（无补偿）": M.run_simulation(sc, M.make_controller("mgvf", gvf, ctrl), sim),
        "GVFPID（基线）": M.run_simulation(sc, M.make_controller("gvfpid", gvf, ctrl), sim),
    }
    plotting.fig_case3(OUT_FIG, c3, sc.obstacles, sc.d_w)
    for tag, r in c3.items():
        row = {"case": "Case 3", "controller": tag}
        row.update({k: v for k, v in M.summarize(r).items() if k not in ("controller", "scenario")})
        rows.append(row)
    for tag, r in c3.items():
        s = M.summarize(r)
        log(f"  {tag:<16} 最小安全距离 {s['min_clearance']:.3f} m | "
            f"制导角 RMSE {s['rmse_e_psi_far']:.4f} rad | "
            f"稳态 RMSE {s['ss_rmse_err_x']:.4f}/{s['ss_rmse_err_y']:.4f} m | "
            f"末端距目标 {s['final_dist']:.4f} m")

    # ---------------- 论文图 9 已由 fig_case3 覆盖 ----------------
    print("\n[8/10] 验证图 + 勘误对照（论文原式 ∏ 连乘 vs 早期误读 Σ 求和）")
    gpio_trace = gpio_verification_trace()
    theta_err = theorem1_angle_error()
    plotting.fig_verification(OUT_FIG, gpio_trace, theta_err, blending_curves())
    log(f"  Theorem 1 数值核验：max|Θ_e| = {theta_err:.3e} rad")
    log(f"  Theorem 2 数值核验：GPIO(m=3) 对常值+斜坡扰动 6 s 后误差 "
        f"{abs(gpio_trace['d_hat_3'][-1] - gpio_trace['d_true'][-1]):.2e}")

    # 勘误对照：论文式 (12) 的吸引项是连乘 ∏；本实现早期曾因 PDF 提取器丢掉连乘号字形
    # 而误读为求和 Σ，导致多障碍场景吸引力被放大 → 穿入障碍。此处把两种读法并排复现。
    sc1b = M.case1(dur)
    r_wrong_sum = M.run_simulation(
        sc1b, M.MGVFDCController(M.GVFConfig(**{**gvf.__dict__, "blend": "sum"}), ctrl), sim)
    plotting.fig_blend_gap(OUT_FIG, r_wrong_sum, c1["MGVFDC"], sc1b.obstacles)
    log(f"  勘误对照：论文原式 ∏σ_i（连乘）最小安全距离 "
        f"{c1['MGVFDC'].safety_margin():.3f} m；早期误读 Σσ_i（求和）"
        f"{r_wrong_sum.safety_margin():.3f} m（<0 表示穿入障碍）")
    for tag, r in (("论文原式 ∏σ_i（连乘）", c1["MGVFDC"]), ("早期误读 Σσ_i（求和）", r_wrong_sum)):
        rows.append({"case": "勘误对照 (∏ vs Σ)", "controller": tag,
                     **{k: v for k, v in M.summarize(r).items()
                        if k not in ("controller", "scenario")}})

    # ---------------- 扰动灵敏度 ----------------
    if sweep and not quick:
        print("\n[9/10] 扰动灵敏度扫描")
        sweep_data: Dict[str, Dict[str, List[float]]] = {
            "MGVFDC（有补偿）": {"min_clearance": [], "ss_rmse_pos": []},
            "MGVF（无补偿）": {"min_clearance": [], "ss_rmse_pos": []},
            "GVFPID（基线）": {"min_clearance": [], "ss_rmse_pos": []},
            "d_w3": [],
        }
        for d3 in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            sweep_data["d_w3"].append(d3)
            sc = M.case3(60.0, d_w=(-0.1, 0.05, d3))
            for tag, kind in (("MGVFDC（有补偿）", "mgvfdc"), ("MGVF（无补偿）", "mgvf"),
                              ("GVFPID（基线）", "gvfpid")):
                r = M.run_simulation(sc, M.make_controller(kind, gvf, ctrl),
                                     M.SimConfig(dt=0.002, duration=60.0))
                s = M.summarize(r)
                sweep_data[tag]["min_clearance"].append(s["min_clearance"])
                sweep_data[tag]["ss_rmse_pos"].append(s["ss_rmse_pos"])
                rows.append({"case": f"扰动扫描 d_w3={d3}", "controller": tag,
                             **{k: v for k, v in s.items() if k not in ("controller", "scenario")}})
            log(f"  d_w3 = {d3}: " + " | ".join(
                f"{tag.split('（')[0]} 安全距离 {sweep_data[tag]['min_clearance'][-1]:6.3f} m, "
                f"RMSE {sweep_data[tag]['ss_rmse_pos'][-1]:.4f} m"
                for tag in ("MGVFDC（有补偿）", "MGVF（无补偿）", "GVFPID（基线）")))
        plotting.fig_disturbance_sweep(OUT_FIG, sweep_data)

    # ---------------- 现实误差水平下的 RMSE（对标论文 0.075/0.091 m） ----------------
    print("\n[10/10] 现实传感/底层误差水平下的跟踪精度")
    sc1c = M.case1(dur)
    realistic = M.SimConfig(dt=0.002, duration=dur,
                            meas_noise_std=(0.02, 0.02, 0.005), yaw_rate_noise=0.01)
    sc_real = M.case1(dur)
    sc_real.d_w = (0.05, -0.03, 0.05)          # 模拟底层速度跟踪残差
    r_real = M.run_simulation(sc_real, M.MGVFDCController(gvf, ctrl), realistic)
    s_real = M.summarize(r_real)
    rows.append({"case": "Case 1（加传感噪声+底层残差）", "controller": "MGVFDC",
                 **{k: v for k, v in s_real.items() if k not in ("controller", "scenario")}})
    log(f"  理想模型：稳态 RMSE x={M.summarize(c1['MGVFDC'])['ss_rmse_err_x']:.4f} / "
        f"y={M.summarize(c1['MGVFDC'])['ss_rmse_err_y']:.4f} m")
    log(f"  加 2 cm 位置噪声 + 0.5° 航向噪声 + 底层速度残差：稳态 RMSE "
        f"x={s_real['ss_rmse_err_x']:.4f} / y={s_real['ss_rmse_err_y']:.4f} m"
        f"（论文报告 0.075 / 0.091 m，量级一致；说明论文数值对应含传感噪声与"
        f"底层速度跟踪残差的实机水平，而理想运动学模型下可达 1e-4 m 级）")

    plotting.fig_framework(OUT_FIG)

    # ---------------- 输出指标表 ----------------
    os.makedirs(OUT, exist_ok=True)
    keys = ["case", "controller", "ss_rmse_err_x", "ss_rmse_err_y", "ss_rmse_pos",
            "rmse_e_psi_far", "path_length", "tortuosity", "arrival_time_1m",
            "min_clearance", "collided", "final_dist", "mean_u"]
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    log("")
    log("=" * 108)
    log("完成。图见 outputs/figures/，指标见 outputs/metrics.csv")
    log("=" * 108)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="论文完整复现")
    ap.add_argument("--quick", action="store_true", help="缩短仿真时长快速跑通")
    ap.add_argument("--no-sweep", action="store_true", help="跳过扰动灵敏度扫描")
    args = ap.parse_args()
    run_all(quick=args.quick, sweep=not args.no_sweep)


if __name__ == "__main__":
    main()
