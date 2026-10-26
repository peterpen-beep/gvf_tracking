"""论文复现的演示视频生成（MP4）。

本脚本是**独立的可视化工具**，不修改 `src/mgvfdc/` 中的任何实现：
它直接调用仿真引擎跑出轨迹，再把轨迹逐帧画成视频。

用法：
    python tools/make_video.py                     # 生成全部 3 段视频
    python tools/make_video.py --only case3        # 只生成某一段
    python tools/make_video.py --preview           # 只渲染前 30 帧（快速自检）
    python tools/make_video.py --fps 30 --dt-frame 0.08 --crf 20

产物：
    outputs/videos/case1_panorama.mp4         Case 1 全景（6 障碍 + 动态目标，MGVFDC）
    outputs/videos/case3_comparison.mp4       Case 3 三方法并排对比（含常值扰动）
    outputs/videos/fig5_chi_sweep.mp4         Fig.5 的 χ 扫描对比
    outputs/videos/<name>_poster.png          每段视频的 2×2 关键帧拼图（便于快速预览）
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径与依赖
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

VENDOR = os.path.join(ROOT, ".vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)          # 工程内安装的 imageio-ffmpeg（自带 ffmpeg 二进制）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

import mgvfdc as M
from mgvfdc.sim import SimResult

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
    "font.size": 9,
})

OUT_DIR = os.path.join(ROOT, "outputs", "videos")

MCOLOR = {"MGVFDC": "#1f77b4", "MGVF": "#d62728", "GVFPID": "#2ca02c"}
CHI_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


# ---------------------------------------------------------------------------
# ffmpeg 定位
# ---------------------------------------------------------------------------
def ffmpeg_path() -> str | None:
    """返回可用的 ffmpeg 可执行文件路径（优先用工程内 imageio-ffmpeg 自带的）。"""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    from shutil import which
    return which("ffmpeg")


def make_writer(path: str, fps: int, crf: int):
    """优先 MP4（libx264），无 ffmpeg 时退回 GIF。"""
    exe = ffmpeg_path()
    if exe and path.lower().endswith(".mp4"):
        matplotlib.rcParams["animation.ffmpeg_path"] = exe
        return FFMpegWriter(fps=fps, codec="libx264",
                            extra_args=["-crf", str(crf), "-pix_fmt", "yuv420p",
                                        "-movflags", "+faststart"]), path
    gif = os.path.splitext(path)[0] + ".gif"
    print(f"  [warn] 未找到 ffmpeg，改为输出 GIF：{os.path.basename(gif)}")
    return PillowWriter(fps=fps), gif


# ---------------------------------------------------------------------------
# 通用绘图组件
# ---------------------------------------------------------------------------
def field_background(ax, obstacles_pos, obstacles_rho, target_pos, cfg, n: int = 17,
                     xlim=None, ylim=None):
    """在给定范围内画 GVF 单位向量场（静态障碍场景只需画一次）。"""
    x0, x1 = xlim
    y0, y1 = ylim
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            F, _ = M.resultant_field(np.array([X[i, j], Y[i, j]]), target_pos,
                                     obstacles_pos, obstacles_rho, cfg)
            nrm = np.linalg.norm(F)
            if nrm > 1e-12:
                U[i, j], V[i, j] = F[0] / nrm, F[1] / nrm
    ax.quiver(X, Y, U, V, color="#8fb8de", alpha=0.55, width=0.0022,
              scale=32, zorder=1, pivot="mid")


def robot_triangle(x: float, y: float, psi: float, size: float = 1.1) -> Polygon:
    """机器人标记：带朝向的三角形。"""
    pts = np.array([[1.0, 0.0], [-0.55, 0.5], [-0.55, -0.5]]) * size
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T + np.array([x, y])
    return Polygon(pts, closed=True, fc="#111111", ec="#ffcc00", lw=1.2, zorder=7)


def draw_obstacles(ax, scenario, t: float):
    """画障碍物本体与安全圆（ρ_b + ρ_o）。"""
    for ob in scenario.obstacles:
        p = ob.position(t)
        ax.add_patch(Circle(p, ob.radius, fc="#c8c8c8", ec="#555555", lw=1.0, zorder=3))
        ax.add_patch(Circle(p, ob.radius + scenario_rho_b, fc="none", ec="#d62728",
                            ls="--", lw=1.0, alpha=0.75, zorder=3))


def arena_limits(results: Sequence[SimResult], scenario, margin: float = 4.0):
    xs = np.concatenate([r.x for r in results] + [r.tgt_x for r in results])
    ys = np.concatenate([r.y for r in results] + [r.tgt_y for r in results])
    for ob in scenario.obstacles:
        p = ob.position(0.0)
        xs = np.append(xs, p[0])
        ys = np.append(ys, p[1])
    return (xs.min() - margin, xs.max() + margin), (ys.min() - margin, ys.max() + margin)


scenario_rho_b = 1.0        # 与 GVFConfig 默认一致；主流程会重新赋值


# ---------------------------------------------------------------------------
# 渲染主体
# ---------------------------------------------------------------------------
def render(results: List[Tuple[str, SimResult]], scenario, path: str, title: str,
           fps: int = 25, dt_frame: float = 0.1, crf: int = 22,
           preview: int = 0, poster: bool = True) -> str:
    """把一条或多条轨迹渲染成视频（并排 N 个战场视图 + 右侧时序曲线）。"""
    n_agent = len(results)
    assert n_agent in (1, 2, 3), "并排视图支持 1~3 个"
    dt_sim = results[0][1].dt
    dur = results[0][1].t[-1]
    stride = max(1, int(round(dt_frame / dt_sim)))
    frames = list(range(0, len(results[0][1].t), stride))
    if preview:
        frames = frames[:preview]
    print(f"  {os.path.basename(path)}: {len(frames)} 帧 × {n_agent} 视图 "
          f"（视频时长 {len(frames)/fps:.1f} s）")

    (x0, x1), (y0, y1) = arena_limits([r for _, r in results], scenario)
    fig = plt.figure(figsize=(13.2 if n_agent == 1 else 13.6,
                              6.4 if n_agent == 1 else 5.4), dpi=100)
    gs = fig.add_gridspec(3, n_agent + 1, width_ratios=[1.0] * n_agent + [1.35],
                          hspace=0.45, wspace=0.22)

    arenas = []
    for i in range(n_agent):
        ax = fig.add_subplot(gs[:, i])
        arenas.append(ax)

    # ---------------- 各智能体视图的静态部分 ----------------
    rho_b = M.GVFConfig().rho_b
    global scenario_rho_b
    scenario_rho_b = rho_b
    obs0 = [ob.position(0.0) for ob in scenario.obstacles]
    rho_o = [ob.radius for ob in scenario.obstacles]
    tgt0 = np.array([results[0][1].tgt_x[0], results[0][1].tgt_y[0]])

    artists = []
    for ax, (label, res) in zip(arenas, results):
        if scenario.obstacles:      # 静态障碍场景：场背景只画一次
            field_background(ax, obs0, rho_o, tgt0, M.GVFConfig(), n=15,
                             xlim=(x0, x1), ylim=(y0, y1))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        if ax is arenas[0]:
            ax.set_ylabel("y [m]")
        ax.grid(alpha=0.25, zorder=0)

        # 目标轨迹与机器人轨迹
        tgt_line, = ax.plot([], [], "-", color="#ffb300", lw=1.6, alpha=0.9, zorder=4)
        robot_line, = ax.plot([], [], "-", color=MCOLOR.get(label.split("（")[0], "#111111"),
                                                             lw=2.2, alpha=0.95, zorder=6)
        obs_patches, obs_safety = [], []
        for ob in scenario.obstacles:
            oc = Circle(ob.position(0.0), ob.radius, fc="#c8c8c8", ec="#555555",
                        lw=1.0, zorder=3)
            sc = Circle(ob.position(0.0), ob.radius + rho_b, fc="none", ec="#d62728",
                        ls="--", lw=1.0, alpha=0.75, zorder=3)
            ax.add_patch(oc)
            ax.add_patch(sc)
            obs_patches.append(oc)
            obs_safety.append(sc)
        tgt_mark, = ax.plot([], [], "*", color="#ffb300", ms=17, mec="#7a5200", zorder=8)
        bot = robot_triangle(0.0, 0.0, 0.0)
        ax.add_patch(bot)
        # 参考航向（ψ_r）与实际航向
        ref_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                                lw=1.4, ls="--"))
        head_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4))
        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8.5,
                       bbox=dict(fc="white", ec="#cccccc", alpha=0.85, pad=2.5), zorder=9)
        ax.set_title(label, fontsize=10)
        artists.append(dict(ax=ax, tgt_line=tgt_line, robot_line=robot_line,
                            obs_patches=obs_patches, obs_safety=obs_safety,
                            tgt_mark=tgt_mark, bot=bot, ref=ref_arrow, head=head_arrow,
                            info=info, res=res))

    # ---------------- 右侧时序曲线 ----------------
    tl = [ax for ax in arenas]
    series_axes = [fig.add_subplot(gs[i, n_agent]) for i in range(3)]
    tgrid = results[0][1].t

    # (1) 距目标距离
    ax1 = series_axes[0]
    dist_lines = []
    for (label, res) in results:
        ln, = ax1.plot([], [], lw=1.4, color=MCOLOR.get(label.split("（")[0], "#333333"),
                                                       label=label)
        dist_lines.append(ln)
    ax1.set_ylabel("距目标 [m]")
    ax1.set_ylim(0, max(1.0, float(np.max([r.dist for _, r in results])) * 1.05))
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.set_title("跟踪误差", fontsize=9)

    # (2) 最小安全距离
    ax2 = series_axes[1]
    clr_lines = []
    for (label, res) in results:
        ln, = ax2.plot([], [], lw=1.4, color=MCOLOR.get(label.split("（")[0], "#333333"))
        clr_lines.append(ln)
    ax2.axhline(rho_b, color="k", ls="--", lw=1.1)
    ax2.text(tgrid[0], rho_b + 0.05, " ρ_b（碰撞判据）", fontsize=7)
    ax2.axhline(0.0, color="#d62728", ls=":", lw=1.0)
    if scenario.obstacles:
        cmin = float(min(np.min(r.clearance) for _, r in results))
        cmax = float(max(np.max(r.clearance) for _, r in results))
        ax2.set_ylim(min(cmin - 0.5, -0.5), min(cmax * 1.1, 12.0))
    ax2.set_ylabel("最小安全距离 [m]")
    ax2.set_title("避障裕度", fontsize=9)

    # (3) 纵向速度 u
    ax3 = series_axes[2]
    u_lines = []
    for (label, res) in results:
        ln, = ax3.plot([], [], lw=1.2, color=MCOLOR.get(label.split("（")[0], "#333333"))
        u_lines.append(ln)
    ax3.set_ylabel("速度 u [m/s]")
    ax3.set_xlabel("t [s]")
    ax3.set_ylim(min(-0.2, float(min(np.min(r.u) for _, r in results)) * 1.05),
                 float(max(np.max(r.u) for _, r in results)) * 1.1)
    ax3.set_title("控制输入", fontsize=9)

    cursors = [ax.axvline(0.0, color="#888888", lw=0.9, ls="-") for ax in series_axes]
    for ax in series_axes:
        ax.set_xlim(tgrid[0], tgrid[-1])
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12.5, y=0.985)

    # ---------------- 帧循环 ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    writer, final_path = make_writer(path, fps, crf)
    poster_idx = set(np.linspace(0, len(frames) - 1, 4).astype(int).tolist()) if poster else set()
    posters: List[str] = []

    with writer.saving(fig, final_path, dpi=100):
        for fidx, k in enumerate(frames):
            t = float(tgrid[k])
            for ad in artists:
                res = ad["res"]
                for ip, (oc, sc, ob) in enumerate(zip(ad["obs_patches"], ad["obs_safety"],
                                                      scenario.obstacles)):
                    p = ob.position(t)
                    oc.center = (float(p[0]), float(p[1]))
                    sc.center = (float(p[0]), float(p[1]))
                ad["tgt_line"].set_data(res.tgt_x[:k + 1], res.tgt_y[:k + 1])
                ad["robot_line"].set_data(res.x[:k + 1], res.y[:k + 1])
                ad["tgt_mark"].set_data([res.tgt_x[k]], [res.tgt_y[k]])
                ad["bot"].set_xy(robot_triangle(res.x[k], res.y[k], res.psi[k]).get_xy())
                x, y, psi = res.x[k], res.y[k], res.psi[k]
                xr, yr = res.psi_r[k], res.psi_r[k]
                L = 2.4
                ad["ref"].set_position((x, y))
                ad["ref"].xy = (x + L * np.cos(res.psi_r[k]), y + L * np.sin(res.psi_r[k]))
                ad["head"].set_position((x, y))
                ad["head"].xy = (x + L * np.cos(psi), y + L * np.sin(psi))
                clr = float(np.min(res.clearance[:, k])) if res.n_obs else float("inf")
                ad["info"].set_text(
                    f"t = {t:5.1f} s\n距目标 {res.dist[k]:5.2f} m\n"
                    f"安全裕度 {clr:5.2f} m\n"
                    f"e_ψ {res.e_psi[k]:+.2f} rad")

            for ln, (_, res) in zip(dist_lines, results):
                ln.set_data(res.t[:k + 1], res.dist[:k + 1])
            for ln, (_, res) in zip(clr_lines, results):
                if res.n_obs:
                    ln.set_data(res.t[:k + 1], res.clearance.min(axis=0)[:k + 1])
            for ln, (_, res) in zip(u_lines, results):
                ln.set_data(res.t[:k + 1], res.u[:k + 1])
            for c in cursors:
                c.set_xdata([t, t])

            writer.grab_frame()
            if fidx in poster_idx:
                p = os.path.join(OUT_DIR, f"_poster_{os.path.basename(final_path)}_{fidx}.png")
                fig.savefig(p, dpi=90)
                posters.append(p)
            if (fidx + 1) % 100 == 0:
                print(f"    ... {fidx + 1}/{len(frames)} 帧")

    plt.close(fig)
    print(f"  [video] {final_path}  ({os.path.getsize(final_path)/1048576:.1f} MB)")

    if posters:
        montage(posters, os.path.splitext(final_path)[0] + "_poster.png")
        for p in posters:
            os.remove(p)
    return final_path


def montage(paths: Sequence[str], out: str) -> None:
    """把 4 张关键帧拼成 2×2 预览图。"""
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    canvas = Image.new("RGB", (w * 2 + 12, h * 2 + 12), "white")
    for idx, im in enumerate(ims[:4]):
        canvas.paste(im, ((idx % 2) * (w + 12), (idx // 2) * (h + 12)))
    canvas.save(out)
    print(f"  [poster] {out}")


# ---------------------------------------------------------------------------
# 三段场景
# ---------------------------------------------------------------------------
def build_case1():
    sc = M.case1(120.0)
    res = M.run_simulation(sc, M.MGVFDCController(M.GVFConfig(), M.GuidanceConfig()),
                           M.SimConfig(dt=0.002, duration=120.0))
    return [("MGVFDC", res)], sc, "Case 1：动态目标 + 6 个静态障碍（MGVFDC 完整方法）"


def build_case3():
    sc = M.case3(120.0)
    gvf, ctrl = M.GVFConfig(), M.GuidanceConfig()
    sim = M.SimConfig(dt=0.002, duration=120.0)
    res = [("MGVFDC（有补偿）", M.run_simulation(sc, M.MGVFDCController(gvf, ctrl), sim))]
    res.append(("MGVF（无补偿）", M.run_simulation(sc, M.make_controller("mgvf", gvf, ctrl), sim)))
    res.append(("GVFPID（基线）", M.run_simulation(sc, M.make_controller("gvfpid", gvf, ctrl), sim)))
    return res, sc, (f"Case 3：窄门障碍 + 常值扰动 d_w = {tuple(sc.d_w)} —— 三方法并排对比")


def build_fig5():
    sc = M.fig5_scenario(45.0)
    sim = M.SimConfig(dt=0.002, duration=45.0)
    out = []
    for chi in (0.1, 0.25, 0.5, 0.75, 0.9):
        r = M.run_simulation(sc, M.MGVFDCController(M.GVFConfig(chi=chi),
                                                    M.GuidanceConfig()), sim)
        out.append((f"χ={chi}", r))
    return out, sc, "Fig.5 场景：可调融合参数 χ 对避障距离的微调（5 个 χ 并排）"


def render_chi_sweep(results, scenario, path: str, title: str,
                     fps: int = 25, dt_frame: float = 0.08, crf: int = 22,
                     preview: int = 0) -> str:
    """χ 扫描专用：轨迹重叠 + 安全距离曲线 + 运行中的最小安全距离柱状图。"""
    dt_sim = results[0][1].dt
    stride = max(1, int(round(dt_frame / dt_sim)))
    tgrid = results[0][1].t
    frames = list(range(0, len(tgrid), stride))
    if preview:
        frames = frames[:preview]
    print(f"  {os.path.basename(path)}: {len(frames)} 帧（视频时长 {len(frames)/fps:.1f} s）")

    (x0, x1), (y0, y1) = arena_limits([r for _, r in results], scenario, margin=3.0)
    fig = plt.figure(figsize=(12.6, 6.0), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.24)
    ax = fig.add_subplot(gs[:, 0])
    obs0 = [ob.position(0.0) for ob in scenario.obstacles]
    rho_o = [ob.radius for ob in scenario.obstacles]
    tgt0 = np.array([results[0][1].tgt_x[0], results[0][1].tgt_y[0]])
    field_background(ax, obs0, rho_o, tgt0, M.GVFConfig(), n=15,
                     xlim=(x0, x1), ylim=(y0, y1))
    for ob in scenario.obstacles:
        ax.add_patch(Circle(ob.position(0.0), ob.radius, fc="#c8c8c8", ec="#555555", zorder=3))
        ax.add_patch(Circle(ob.position(0.0), ob.radius + scenario_rho_b, fc="none",
                            ec="#d62728", ls="--", lw=1.0, alpha=0.75, zorder=3))
    ax.plot([], [])
    trails, bots = [], []
    for (label, res), c in zip(results, CHI_COLORS):
        ln, = ax.plot([], [], "-", color=c, lw=2.0, label=label, zorder=6)
        trails.append(ln)
        bots.append(ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2)))
    ax.plot(tgt0[0], tgt0[1], "*", color="#ffb300", ms=18, mec="#7a5200", zorder=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("轨迹：χ 越大，越贴障碍物边界", fontsize=10)

    axc = fig.add_subplot(gs[0, 1])
    clr_lines = []
    for (label, res), c in zip(results, CHI_COLORS):
        ln, = axc.plot([], [], lw=1.5, color=c, label=label)
        clr_lines.append(ln)
    axc.axhline(scenario_rho_b, color="k", ls="--", lw=1.2)
    axc.axhline(0.0, color="#d62728", ls=":", lw=1.0)
    axc.set_ylim(0.5, float(max(np.max(r.clearance) for _, r in results)) * 1.1)
    axc.set_ylabel("到障碍边界的距离 [m]")
    axc.set_xlim(tgrid[0], tgrid[-1])
    axc.grid(alpha=0.3)
    axc.set_title("瞬时安全距离曲线", fontsize=10)
    cur = axc.axvline(0.0, color="#888888", lw=0.9)

    axb = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(results))
    bars = axb.bar(xs, np.zeros(len(results)), color=CHI_COLORS, alpha=0.85)
    axb.axhline(scenario_rho_b, color="k", ls="--", lw=1.2)
    axb.set_xticks(xs)
    axb.set_xticklabels([lab for lab, _ in results], fontsize=8.5)
    axb.set_ylim(0.0, float(max(np.max(r.clearance) for _, r in results)) * 1.1)
    axb.set_ylabel("当前最小安全距离 [m]")
    axb.set_title("各 χ 的历史最小安全距离（运行中滚增）", fontsize=10)
    txt = axb.text(0.02, 0.9, "", transform=axb.transAxes, fontsize=8.5)

    fig.suptitle(title, fontsize=12.5, y=0.98)

    os.makedirs(OUT_DIR, exist_ok=True)
    writer, final_path = make_writer(path, fps, crf)
    with writer.saving(fig, final_path, dpi=100):
        for fidx, k in enumerate(frames):
            t = float(tgrid[k])
            for ln, bot, (label, res), c in zip(trails, bots, results, CHI_COLORS):
                ln.set_data(res.x[:k + 1], res.y[:k + 1])
                bot.set_position((res.x[k], res.y[k]))
                bot.xy = (res.x[k] + 1.6 * np.cos(res.psi[k]),
                          res.y[k] + 1.6 * np.sin(res.psi[k]))
            run_min = []
            for ln, (label, res) in zip(clr_lines, results):
                clr = res.clearance[0]
                ln.set_data(res.t[:k + 1], clr[:k + 1])
                run_min.append(float(np.min(clr[:k + 1])))
            for b, v in zip(bars, run_min):
                b.set_height(v)
            txt.set_text(f"t = {t:4.1f} s")
            cur.set_xdata([t, t])
            writer.grab_frame()
            if (fidx + 1) % 100 == 0:
                print(f"    ... {fidx + 1}/{len(frames)} 帧")
    plt.close(fig)
    print(f"  [video] {final_path}  ({os.path.getsize(final_path)/1048576:.1f} MB)")
    return final_path


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="论文复现演示视频")
    ap.add_argument("--only", choices=["case1", "case3", "fig5"], default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dt-frame", type=float, default=0.1, help="每帧对应的仿真时间 [s]")
    ap.add_argument("--crf", type=int, default=22, help="H.264 质量（越小越清晰、体积越大）")
    ap.add_argument("--preview", action="store_true", help="只渲染前 30 帧自检")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prev = 30 if args.preview else 0
    jobs = [args.only] if args.only else ["case1", "case3", "fig5"]

    for job in jobs:
        print(f"\n=== 生成 {job} ===")
        if job == "case1":
            res, sc, title = build_case1()
            render(res, sc, os.path.join(OUT_DIR, "case1_panorama.mp4"), title,
                   fps=args.fps, dt_frame=args.dt_frame, crf=args.crf, preview=prev)
        elif job == "case3":
            res, sc, title = build_case3()
            render(res, sc, os.path.join(OUT_DIR, "case3_comparison.mp4"), title,
                   fps=args.fps, dt_frame=max(args.dt_frame, 0.12), crf=args.crf, preview=prev)
        else:
            res, sc, title = build_fig5()
            render_chi_sweep(res, sc, os.path.join(OUT_DIR, "fig5_chi_sweep.mp4"), title,
                             fps=args.fps, dt_frame=max(args.dt_frame, 0.06), crf=args.crf,
                             preview=prev)
    print("\n完成，视频位于 outputs/videos/")


if __name__ == "__main__":
    main()
