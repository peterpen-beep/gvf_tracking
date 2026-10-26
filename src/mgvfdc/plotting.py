"""绘图模块：逐一复现论文 Fig. 3~Fig. 9 对应的图，并给出验证与消融图。

所有图保存到 outputs/figures/。中文字体自动选择（Windows 下微软雅黑/黑体）。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow

from .config import GVFConfig
from .gvf import (attractive_field, beta_triplet, blending_function,
                  integrate_field_line, resultant_field)
from .metrics import summarize
from .plant import Obstacle
from .sim import SimResult

# ----------------------------------------------------------------------------
# 全局绘图风格
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
          "#17becf", "#e377c2"]


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save(fig, outdir: str, name: str) -> str:
    _ensure(outdir)
    p = os.path.join(outdir, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  [fig] {p}")
    return p


# ----------------------------------------------------------------------------
# 图 0：控制框架框图（对应论文 Fig. 2）
# ----------------------------------------------------------------------------
def fig_framework(outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf2fb", ec="#2c6fbb"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, zorder=3)

    def arrow(x1, y1, x2, y2, text="", style="-|>", color="#333333", off=0.12):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + off, text, ha="center",
                    fontsize=8.5, color=color)

    box(0.2, 3.3, 2.1, 1.1, "① 改进 GVF 向量场\n（式 (4)-(20)）")
    box(3.1, 3.3, 2.3, 1.1, "② 制导控制器\nPD 转向 + 滑模趋近\n（式 (21)-(33)）", fc="#fdeaea", ec="#c0392b")
    box(6.2, 3.3, 2.0, 1.1, "转向优先策略\n（式 (32)(33)）", fc="#fff7e6", ec="#e08b00")
    box(8.9, 3.3, 1.9, 1.1, "④ 动力学控制器\n（引文[6] 凸MPC）\n四足机器人", fc="#eafaf0", ec="#27ae60")
    box(3.1, 1.2, 2.3, 1.0, "③ GPIO 扰动观测器\n（式 (23)(28)）", fc="#f5eafd", ec="#8e44ad")

    arrow(2.3, 3.85, 3.1, 3.85, "ψ_r（参考航向角）")
    arrow(5.4, 3.85, 6.2, 3.85, "u, r")
    arrow(8.2, 3.85, 8.9, 3.85, "期望速度")
    arrow(4.95, 3.3, 4.35, 2.2, "补偿 d̂")
    arrow(8.9, 3.4, 4.4, 1.7, "测量 ψ, s", color="#8e44ad")
    ax.annotate("", xy=(0.2, 3.6), xytext=(9.8, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=1.3,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(5.0, 0.55, "实线=本文贡献（①②③）　　虚线/绿框=引文 [6] 的既有动力学控制层",
            ha="center", fontsize=9, color="#555555")
    ax.set_title("论文 Fig.2 的无碰撞目标跟踪控制框架（本复现实现的模块标注）", fontsize=12)
    return save(fig, outdir, "fig0_framework.png")


# ----------------------------------------------------------------------------
# 图 1：吸引场对比（论文 Fig. 3）
# ----------------------------------------------------------------------------
def fig_field_comparison(outdir: str, target=(5.0, 5.0), start=(6.5, 6.5)) -> str:
    target = np.asarray(target, dtype=float)
    start = np.asarray(start, dtype=float)
    cfg_fixed = GVFConfig(att_mode="fixed", fixed_p=(1.0, 0.0))
    cfg_var = GVFConfig(att_mode="variable")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, cfg, title in ((axes[0], cfg_fixed, "(a) 经典 GVF：固定属性向量 p（绕远路）"),
                           (axes[1], cfg_var, "(b) 改进 GVF：可变属性向量 p = −η_e（直指目标）")):
        lo, hi = -0.5, 8.0
        xs = np.linspace(lo, hi, 26)
        ys = np.linspace(lo, hi, 26)
        U = np.zeros((len(ys), len(xs)))
        V = np.zeros_like(U)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                F, _ = resultant_field(np.array([x, y]), target, (), (), cfg)
                n = np.linalg.norm(F)
                U[i, j], V[i, j] = (F / n) if n > 1e-12 else (0.0, 0.0)
        ax.streamplot(xs, ys, U, V, density=1.15, color="#8fb8de",
                      linewidth=0.8, arrowsize=0.7, zorder=1)

        # 特殊积分曲线：从分离线区域出发（论文中的 S 点）
        for sx, sy in [(6.5, 6.5), (7.5, 7.5), (4.0, 6.0), (6.0, 4.0), (7.0, 5.0)]:
            line = integrate_field_line(np.array([sx, sy]), target, cfg, steps=6000, ds=0.01)
            ax.plot(line[:, 0], line[:, 1], "-", color="#d62728", lw=1.4, alpha=0.85, zorder=3)

        traj = integrate_field_line(start, target, cfg, steps=20000, ds=0.003)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#111111", lw=2.4, zorder=5,
                label=f"从 S 点出发轨迹（长 {np.sum(np.hypot(*np.diff(traj, axis=0).T)):.2f} m）")
        ax.plot(*start, "o", color="#111111", ms=7, zorder=6)
        ax.annotate("S", start, textcoords="offset points", xytext=(8, 6), fontsize=11)
        ax.plot(*target, "*", color="#ffb300", ms=20, mec="#7a5200", zorder=6)
        ax.annotate("目标 [5,5]", target, textcoords="offset points", xytext=(8, -14), fontsize=9)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("复现论文 Fig.3：改进 1（可变属性向量）把椭圆积分曲线形变为指向目标的直线", fontsize=12)
    return save(fig, outdir, "fig1_field_comparison.png")


# ----------------------------------------------------------------------------
# 图 1b：闭环"绕远路"量化（论文 Fig. 3 / Remark 5 的闭环版本）
# ----------------------------------------------------------------------------
def fig_detour_closedloop(outdir: str, res_fixed: SimResult, res_var: SimResult) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.plot(res_fixed.x, res_fixed.y, "-", color="#d62728", lw=2.2,
            label="固定 p（经典 GVF）")
    ax.plot(res_var.x, res_var.y, "-", color="#2ca02c", lw=2.2,
            label="可变 p = −η_e（改进 1）")
    ax.plot([res_fixed.x[0], 0], [res_fixed.y[0], 0], "k--", lw=1.2, alpha=0.6,
            label="起终点直线（28.284 m）")
    ax.plot(res_fixed.x[0], res_fixed.y[0], "ks", ms=8)
    ax.plot(0, 0, "*", color="#ffb300", ms=20, mec="#7a5200")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) 闭环轨迹：固定 p 绕行圆弧，可变 p 直线到达", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for res, c, tag in ((res_fixed, "#d62728", "固定 p"), (res_var, "#2ca02c", "可变 p")):
        L = float(np.sum(np.hypot(np.diff(res.x), np.diff(res.y))))
        ax.plot(res.t, res.dist, lw=1.8, color=c,
                label=f"{tag}（路径 {L:.2f} m）")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("距目标距离 [m]")
    ax.set_yscale("log")
    ax.set_title("(b) 到目标的距离收敛过程（固定 p 更晚收敛）", fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.suptitle("复现论文 Fig.3 的闭环量化：改进 1 消除了绕远路（28.284 m = 零绕行）", fontsize=12)
    return save(fig, outdir, "fig1b_detour_closedloop.png")


# ----------------------------------------------------------------------------
# 图 2：融合函数（论文 Fig. 4）
# ----------------------------------------------------------------------------
def fig_blending(outdir: str, rho_o: float = 1.0, rho_b: float = 1.0,
                 rho_a: float = 2.1, rho_s: float = 1.1) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cfg0 = GVFConfig(rho_b=rho_b, rho_a=rho_a, rho_s=rho_s)
    _, beta_a, beta_s = beta_triplet(1.0, rho_o, cfg0)
    delta = beta_s - beta_a

    ax = axes[0]
    betas = np.linspace(beta_a, beta_s, 600)
    for chi, c in zip((0.1, 0.25, 0.5, 0.75, 0.9), COLORS):
        vals = [blending_function(b, beta_a, beta_s, chi) for b in betas]
        ax.plot(betas, vals, color=c, lw=1.8, label=f"χ = {chi}")
        bj = beta_a + chi * delta
        ax.plot([bj], [1 - chi], "o", color=c, ms=5)
    ax.set_xlabel(r"$\beta_i$")
    ax.set_ylabel(r"$\xi(\beta_i)=\sigma_i(\beta_i)$")
    ax.set_title(r"(a) 融合函数族（实心点=接缝 $\beta_a+\chi\Delta$ 处取值 $1-\chi$）", fontsize=10.5)
    ax.legend(fontsize=8.5)

    # 以"机器人到障碍中心距离"为横轴重画，更直观
    ax = axes[1]
    d = np.linspace(rho_o + rho_b + rho_s - 0.3, rho_o + rho_b + rho_a + 0.6, 800)
    for chi, c in zip((0.1, 0.25, 0.5, 0.75, 0.9), COLORS):
        vals = []
        for dd in d:
            beta, ba, bs = beta_triplet(float(dd), rho_o, cfg0)
            vals.append(blending_function(beta, ba, bs, chi))
        ax.plot(d, vals, color=c, lw=1.8, label=f"χ = {chi}")
    ax.axvline(rho_o + rho_b + rho_s, color="k", ls=":", lw=1.2)
    ax.axvline(rho_o + rho_b + rho_a, color="k", ls="--", lw=1.2)
    ax.text(rho_o + rho_b + rho_s - 0.02, 0.5, " ρ_s 内：纯斥力", rotation=90, fontsize=8, va="center")
    ax.text(rho_o + rho_b + rho_a + 0.03, 0.5, "激活距离外：纯吸引", rotation=90, fontsize=8, va="center")
    ax.set_xlabel("机器人到障碍物中心的距离 d [m]")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("(b) 以距离表示：远时为 1（纯吸引），近时为 0（纯斥力）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("复现论文 Fig.4：改进 2（可调融合函数）—— χ 调节避障过渡带的形状", fontsize=12)
    return save(fig, outdir, "fig2_blending.png")


# ----------------------------------------------------------------------------
# 图 3：χ 微调避障性能（论文 Fig. 5）
# ----------------------------------------------------------------------------
def fig_chi_tuning(outdir: str, results: Dict[float, SimResult],
                   obstacle: Obstacle, target_xy=(0.0, 0.0)) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    for (chi, res), c in zip(results.items(), COLORS):
        ax.plot(res.x, res.y, "-", color=c, lw=1.8, label=f"χ = {chi}")
        ax.plot(res.x[-1], res.y[-1], "o", color=c, ms=5)
    ax.add_patch(Circle(obstacle.pos0, obstacle.radius, fc="#cccccc", ec="#555555", zorder=1))
    ax.add_patch(Circle(obstacle.pos0, obstacle.radius + 1.0, fc="none", ec="#d62728",
                        ls="--", lw=1.2, zorder=1))
    ax.plot(*results[list(results)[0]].x[:1], "ks", ms=8, label="起点")
    ax.plot(*target_xy, "*", color="#ffb300", ms=20, mec="#7a5200", label="目标")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) 不同 χ 下的避障轨迹（红虚线=安全圆 ρ_b+ρ_o）", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    chis = list(results.keys())
    clear = [results[c].safety_margin() for c in chis]
    pathlen = [float(np.sum(np.hypot(np.diff(results[c].x), np.diff(results[c].y)))) for c in chis]
    ax.plot(chis, clear, "o-", color="#d62728", lw=2, label="最小安全距离 [m]")
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.text(chis[0], 1.02, " ρ_b = 1.0 m（安全判据）", fontsize=8.5)
    ax.set_xlabel("χ")
    ax.set_ylabel("最小安全距离 [m]", color="#d62728")
    ax2 = ax.twinx()
    ax2.plot(chis, pathlen, "s--", color="#1f77b4", lw=1.6, label="路径长度 [m]")
    ax2.set_ylabel("路径长度 [m]", color="#1f77b4")
    ax2.grid(False)
    ax.set_title("(b) χ 增大 -> 贴障碍更近（复现论文 Fig.5 结论）", fontsize=10.5)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="upper right")
    fig.suptitle("复现论文 Fig.5：可调参数 χ 对避障性能的微调作用", fontsize=12)
    return save(fig, outdir, "fig3_chi_tuning.png")


# ----------------------------------------------------------------------------
# 图 4：Case 1 跟踪性能（论文 Fig. 6）
# ----------------------------------------------------------------------------
def fig_case1(outdir: str, res: SimResult, obstacles: Sequence[Obstacle],
              extra_results: Dict[str, SimResult] | None = None) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    for ob in obstacles:
        ax.add_patch(Circle(ob.position(0.0), ob.radius, fc="#cccccc",
                            ec="#555555", zorder=1))
        ax.add_patch(Circle(ob.position(0.0), ob.radius + 1.0, fc="none",
                            ec="#d62728", ls="--", lw=0.9, alpha=0.7, zorder=1))
    for i, (tag, r) in enumerate((extra_results or {}).items()):
        ax.plot(r.x, r.y, "--", color=COLORS[i + 2], lw=1.4, alpha=0.85, label=tag)
    ax.plot(res.tgt_x, res.tgt_y, "-", color="#ffb300", lw=2.2, label="目标轨迹")
    ax.plot(res.x, res.y, "-", color="#111111", lw=2.0, label="MGVFDC 轨迹")
    ax.plot(res.x[0], res.y[0], "ks", ms=8, label="起点")
    ax.plot(res.tgt_x[0], res.tgt_y[0], "*", color="#ffb300", ms=18, mec="#7a5200")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"(a) Case 1 轨迹（最小安全距离 {res.safety_margin():.2f} m > ρ_b=1.0）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    ax.plot(res.t, res.err_x, lw=1.2, label="位置误差 $x_e$")
    ax.plot(res.t, res.err_y, lw=1.2, label="位置误差 $y_e$")
    ax.plot(res.t, res.dist, lw=1.6, color="#111111", label="距目标距离")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("误差 [m]")
    s = summarize(res)
    ax.set_title(f"(b) 跟踪误差（稳态 RMSE x={s['ss_rmse_err_x']:.4f} m, "
                 f"y={s['ss_rmse_err_y']:.4f} m）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("复现论文 Fig.6：Case 1（动态目标 + 静态多障碍）的跟踪性能", fontsize=12)
    return save(fig, outdir, "fig4_case1_tracking.png")


# ----------------------------------------------------------------------------
# 图 5：Case 2 快照对比（论文 Fig. 7）
# ----------------------------------------------------------------------------
def fig_case2(outdir: str, results: Dict[str, SimResult],
              obstacles_at=None, n_snap: int = 5) -> str:
    keys = list(results.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6.4 * len(keys), 5.8), squeeze=False)
    for ax, key in zip(axes[0], keys):
        res = results[key]
        n = len(res.t)
        marks = np.linspace(int(0.12 * n), n - 1, n_snap).astype(int)
        for i, k in enumerate(marks):
            ax.plot(res.x[k], res.y[k], "*", ms=15, color=COLORS[i % len(COLORS)],
                    mec="k", mew=0.5, zorder=6)
            if i < len(marks) - 1:
                k2 = marks[i + 1]
                ax.plot(res.x[k:k2], res.y[k:k2], "-", color=COLORS[i % len(COLORS)],
                        lw=1.8, alpha=0.9, zorder=4)
        ax.plot(res.tgt_x, res.tgt_y, ":", color="#ffb300", lw=1.6, label="目标轨迹")
        ax.plot(res.x, res.y, "-", color="#bbbbbb", lw=0.8, zorder=1)
        if obstacles_at is not None:
            for ob in obstacles_at(0.0):
                ax.add_patch(Circle(ob[0], ob[1], fc="#cccccc", ec="#555555", alpha=0.8))
        clr = res.safety_margin()
        s = summarize(res)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{res.controller}\n最小安全距离 {clr:.2f} m，路径 {s['path_length']:.1f} m",
                     fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("复现论文 Fig.7：Case 2（动态目标 + 动态障碍）跟踪状态快照对比（★为等间隔时刻）",
                 fontsize=12)
    return save(fig, outdir, "fig5_case2_snapshots.png")


# ----------------------------------------------------------------------------
# 图 6：安全距离指标（论文 Fig. 8）
# ----------------------------------------------------------------------------
def fig_safety(outdir: str, res: SimResult, rho_b: float = 1.0) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    n = res.n_obs
    for i in range(n):
        ax.plot(res.t, res.clearance[i], lw=1.5, color=COLORS[i % len(COLORS)],
                label=f"障碍 {i+1} (min {res.clearance[i].min():.2f} m)")
    ax.axhline(rho_b, color="k", ls="--", lw=1.6)
    ax.text(res.t[0], rho_b + 0.05, f" ρ_b = {rho_b} m（碰撞判据）", fontsize=9)
    ax.axhline(0.0, color="#d62728", ls=":", lw=1.2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"安全距离指标 $d_i-\rho_{o,i}$ [m]")
    ax.set_ylim(min(-0.5, float(res.clearance.min()) - 0.3), None)
    ax.set_title(f"复现论文 Fig.8：各障碍物的安全距离指标（全程 > ρ_b 即无碰撞）\n"
                 f"{res.controller} @ {res.scenario}", fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    return save(fig, outdir, "fig6_safety_distance.png")


# ----------------------------------------------------------------------------
# 图 7：Case 3 抗扰对比（论文 Fig. 9）
# ----------------------------------------------------------------------------
def fig_case3(outdir: str, results: Dict[str, SimResult],
              obstacles: Sequence[Obstacle], d_w) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for ob in obstacles:
        ax.add_patch(Circle(ob.pos0, ob.radius, fc="#cccccc", ec="#555555"))
        ax.add_patch(Circle(ob.pos0, ob.radius + 1.0, fc="none", ec="#d62728",
                            ls="--", lw=0.9, alpha=0.7))
    for (tag, res), c in zip(results.items(), COLORS):
        style = "-" if "MGVFDC" in tag else ("--" if "MGVF" in tag else ":")
        ax.plot(res.x, res.y, style, color=c, lw=2.0,
                label=f"{tag}（安全距离 {res.safety_margin():.2f} m）")
    first = list(results.values())[0]
    ax.plot(first.tgt_x, first.tgt_y, "-", color="#ffb300", lw=1.6, alpha=0.8, label="目标轨迹")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"(a) 轨迹对比（$d_w$ = {tuple(round(v,3) for v in d_w)}）", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for (tag, res), c in zip(results.items(), COLORS):
        ax.plot(res.t, res.dist, lw=1.6, color=c, label=tag)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("距目标距离 [m]")
    ax.set_yscale("log")
    ax.set_title("(b) 跟踪误差收敛过程", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[2]
    for (tag, res), c in zip(results.items(), COLORS):
        ax.plot(res.t, res.clearance.min(axis=0), lw=1.8, color=c, label=tag)
    ax.axhline(1.0, color="k", ls="--", lw=1.4)
    ax.text(res.t[0], 1.05, " ρ_b = 1.0 m（论文安全判据）", fontsize=8.5)
    ax.axhline(0.0, color="#d62728", ls=":", lw=1.4)
    ax.text(res.t[0], 0.08, " 0 m（障碍物边界）", fontsize=8.5, color="#d62728")
    ax.axhspan(-1.5, 1.0, color="#d62728", alpha=0.08)
    ax.set_ylim(-1.2, 6.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("最小安全距离（对所有障碍取最小）[m]")
    ax.set_title("(c) 安全裕度：红色阴影=违反 ρ_b 判据", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("复现论文 Fig.9：含不确定扰动环境下的抗扰对比（MGVFDC vs MGVF vs GVFPID）", fontsize=12)
    return save(fig, outdir, "fig7_case3_disturbance.png")


# ----------------------------------------------------------------------------
# 图 8：理论性质与观测器验证
# ----------------------------------------------------------------------------
def fig_verification(outdir: str, gpio_trace: Dict[str, np.ndarray],
                     theta_err: float, blend_chi_curve: Dict[float, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.bar(["Theorem 1\n夹角误差"], [-np.log10(max(theta_err, 1e-18))], color="#2ca02c")
    ax.set_ylabel(r"$-\log_{10}$(误差) [rad]")
    ax.set_title(f"(a) 吸引场指向性：max|Θ_e| = {theta_err:.2e} rad", fontsize=10.5)

    ax = axes[1]
    ax.plot(gpio_trace["t"], gpio_trace["d_true"], "k--", lw=1.8, label="真实扰动 $d_{w3}$")
    ax.plot(gpio_trace["t"], gpio_trace["d_hat_2"], lw=1.6, label="GPIO 估计 (m=2)")
    ax.plot(gpio_trace["t"], gpio_trace["d_hat_3"], lw=1.6, label="GPIO 估计 (m=3)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\hat d_{w3}$")
    ax.set_title("(b) Theorem 2：GPIO 对扰动的估计收敛", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[2]
    for (chi, arr), c in zip(blend_chi_curve.items(), COLORS):
        betas, xi, bj, dval = arr
        ax.plot(betas, xi, "-", color=c, lw=1.7, label=f"χ={chi}: ξ")
        ax.axvline(float(bj[0]), color=c, ls=":", lw=1.0, alpha=0.7)
    # ξ' 用右轴单独画（接缝处两侧取值相等即为 C¹ 连续）
    ax2 = ax.twinx()
    first = list(blend_chi_curve.values())[0]
    dxi = np.gradient(first[1], first[0])
    ax2.plot(first[0], dxi, "k--", lw=1.3, label="ξ'(β)（χ=0.1）")
    ax2.set_ylabel("dξ/dβ")
    ax2.grid(False)
    ax.set_xlabel(r"$\beta_i$")
    ax.set_ylabel(r"$\xi(\beta_i)$")
    ax.set_title("(c) 融合函数 C¹ 连续（虚线=接缝；导数曲线无跳变）", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    fig.suptitle("论文理论性质的数值验证", fontsize=12)
    return save(fig, outdir, "fig8_verification.png")


# ----------------------------------------------------------------------------
# 图 9：勘误对照 —— 论文原式（连乘 ∏）vs 我最初误读的（求和 Σ）
# ----------------------------------------------------------------------------
def fig_blend_gap(outdir: str, res_wrong_sum: SimResult, res_paper: SimResult,
                  obstacles: Sequence[Obstacle]) -> str:
    """左：轨迹差异；右：安全裕度。红色=误读的 Σ 版本，绿色=论文原式 ∏ 版本。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for ob in obstacles:
        ax.add_patch(Circle(ob.pos0, ob.radius, fc="#cccccc", ec="#555555"))
        ax.add_patch(Circle(ob.pos0, ob.radius + 1.0, fc="none", ec="#d62728",
                            ls="--", lw=0.9, alpha=0.7))
    ax.plot(res_wrong_sum.x, res_wrong_sum.y, "-", color="#d62728", lw=2,
            label=f"误读：Σσ_i 求和（最小安全距离 {res_wrong_sum.safety_margin():.2f} m）")
    ax.plot(res_paper.x, res_paper.y, "-", color="#2ca02c", lw=2,
            label=f"论文原式：∏σ_i 连乘（{res_paper.safety_margin():.2f} m）")
    ax.plot(res_wrong_sum.tgt_x, res_wrong_sum.tgt_y, ":", color="#ffb300", lw=1.5,
            label="目标轨迹")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) 两种读法的轨迹差异", fontsize=10.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(res_wrong_sum.t, res_wrong_sum.clearance.min(axis=0), lw=1.6, color="#d62728",
            label="误读：Σσ_i 求和")
    ax.plot(res_paper.t, res_paper.clearance.min(axis=0), lw=1.6, color="#2ca02c",
            label="论文原式：∏σ_i 连乘")
    ax.axhline(1.0, color="k", ls="--", lw=1.4)
    ax.axhline(0.0, color="#d62728", ls=":", lw=1.2)
    ax.text(res_paper.t[0], 1.05, " ρ_b = 1.0 m", fontsize=8.5)
    # 显式给出 y 范围，保证"穿入障碍（负值）"清晰可见
    lo = min(-1.0, float(min(res_wrong_sum.clearance.min(), res_paper.clearance.min())) - 0.3)
    hi = max(3.0, float(min(res_paper.clearance.max(), 6.0)) + 0.3)
    ax.set_ylim(lo, hi)
    ax.axhspan(lo, 0.0, color="#d62728", alpha=0.10)
    ax.text(res_paper.t[-1], 0.05 + lo, " 0 m = 障碍物边界", fontsize=8, color="#a00",
            ha="right", va="bottom")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("最小安全距离 [m]")
    ax.set_title("(b) 安全裕度：误读版本会穿入障碍（< 0）", fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.suptitle("勘误对照：论文式 (12) 的吸引项是连乘 ∏（障碍物只削弱吸引，绝不产生吸引）", fontsize=12)
    return save(fig, outdir, "fig9_blend_gap.png")


# ----------------------------------------------------------------------------
# 图 10：扰动灵敏度
# ----------------------------------------------------------------------------
def fig_disturbance_sweep(outdir: str, sweep: Dict[str, Dict[str, List[float]]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    d3 = sweep["d_w3"]

    ax = axes[0]
    for tag, vals in sweep.items():
        if tag == "d_w3":
            continue
        ax.plot(d3, vals["min_clearance"], "o-", lw=1.8, label=tag)
    ax.axhline(1.0, color="k", ls="--", lw=1.4)
    ax.text(d3[0], 1.05, " ρ_b = 1.0 m", fontsize=8.5)
    ax.set_xlabel(r"偏航扰动 $d_{w3}$ [rad/s]")
    ax.set_ylabel("最小安全距离 [m]")
    ax.set_title("(a) 安全裕度随扰动强度的变化", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for tag, vals in sweep.items():
        if tag == "d_w3":
            continue
        ax.semilogy(d3, np.maximum(vals["ss_rmse_pos"], 1e-6), "o-", lw=1.8, label=tag)
    ax.set_xlabel(r"偏航扰动 $d_{w3}$ [rad/s]")
    ax.set_ylabel("稳态位置 RMSE [m]")
    ax.set_title("(b) 跟踪精度随扰动强度的变化", fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.suptitle("扰动灵敏度分析：GPIO 补偿的有效范围（论文仅给出单点 $d_{w3}=0.4$）", fontsize=12)
    return save(fig, outdir, "fig10_disturbance_sweep.png")
