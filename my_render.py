"""my_render.py —— 把 my_run.py 的运行结果画成"论文风格"的图（算法、参数与 my_run 同步）

与 my_run.py 的关系（请务必看这一段）
------------------------------------
1. 循环是"逐行复刻"你 main() 的执行顺序（含"目标在循环末尾更新、gvf 用上一步的目标"、
   "观测器在算控制量之前推进且输入是上一步的 r_pre/u_pre"、`fai_r_dot` 的 ±π 归一化等）。
2. 观测器、增益、参数全部直接 import 你的：`R.gpio / R.init_gpio`，参数见 PARAMS。
3. 向量场：本文件内置了一份与 `my_run.gvf` 逐行一致的实现 `gvf_field(..., normalize=True)`，
   **两处归一化默认开启**，与你 my_run.py 第 13-16、60-63 行的当前状态一致。
   `normalize=False` 可切回"不归一化"版本做对比（那时 |F_a|≈d_e³、|F_r|≈d_o³，
   近障碍时吸引力强几十倍，最小安全裕度会掉到 1.0 以下，违反论文式(3)）。
4. 障碍物位置与 my_run.py:130-135 **保持同步**（改一处要改另一处）。

用法
----
    python my_render.py                 # 出图（默认：无扰动 + 有扰动 两个面板 + 信号 + 末段放大）
    python my_render.py --check         # 用"不归一化"跑一遍，与你 my_run.main() 的打印坐标对比
    python my_render.py --steps 20000
    python my_render.py --no-nodist     # 只画有扰动那一组

产物：outputs/my_render/fig1_trajectory.png / fig2_signals.png / fig3_final_approach.png
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

import my_run as R                       # ← 你的模块：gvf / gpio / init_gpio


# ===========================================================================================
# 一、参数与场景（逐项对应 my_run.py，改的时候两边一起改）
# ===========================================================================================
PARAMS = {                                # 对应 my_run.py:110-119
    'm': 3,
    'omega_o': 8.0,
    'k1': 2.0, 'k2': 0.6,
    'k3': 1.0, 'k4': 0.5,
    'lambda_val': 1.0,
    'epsilon': 1.0,
    'fai_T': 0.5,
    'u_max': 1.5,
}

DT = 0.002                                # my_run.py:122
STEPS = 20000                             # my_run.py:123  （40 s）
START = np.array([0.0, 0.0, 0.0])         # my_run.py:126
TARGET0 = np.array([10.0, 10.0])          # my_run.py:129
TGT_OMEGA = 0.2                           # my_run.py:215 的 0.2 → 半径 2 m ⇒ 切向速度 0.4 m/s
DW = np.array([-0.1, 0.05, 0.4])          # my_run.py:138

# 向量场参数（对应 my_run.py:34-42 的局部变量，这里提出来方便画图与调参）
RHO_B = 1.0                               # robot_safe
RHO_A = 2.1                               # dist_open （避障激活距离）
RHO_S = 1.1                               # dist_end  （最小安全距离）
CHI = 0.5                                 # chis

# 【同步】与 my_run.py:130-135 完全一致（改一处要改另一处！）。布局设计意图：
#   · 三个障碍沿 (0,0)→目标 的直线交错摆放：①在直线右下、②几乎正压在直线上、③再回到右下，
#     形成"右→左→右"的蛇行，能一眼看出切向斥力场 + ∏σ 断电在起作用；
#   · 所有障碍到目标圆心 (10,10) 的距离都 ≥ 6 m（目标圆周半径 2 m），
#     保证"目标本身"不会撞进障碍安全区，否则跟随阶段的安全裕度指标会失真。
#   · 实测（含归一化，40 s）：14.3 s 追上目标、末端距目标 0.00 m、
#     最小安全裕度 1.55 m（> ρ_b=1.0 ✓）。
OBSTACLES = [
    np.array([3.0, 1.6, 0.6]),             # ① 直线右下 → 从上方绕过
    np.array([5.6, 6.9, 0.8]),             # ② 压在直线上 → 反向绕（最大一次变向）
    np.array([8.2, 2.6, 0.6]),             # ③ 又回到右下 → 第二次变向，然后直奔目标
]

OUT_DIR = os.path.join(HERE, "outputs", "my_render")

# 绘图风格
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ===========================================================================================
# 二、向量场：你的 gvf + 【启用】两处归一化
# ===========================================================================================
def gvf_field(robot_pos, target_pos, obstacles, normalize=True, rho_b=RHO_B,
              rho_a=RHO_A, rho_s=RHO_S, chi=CHI):
    """与 my_run.gvf 逐行一致，唯一区别：`normalize=True` 时启用两处归一化。

    normalize=True ：
        F = ∏σ_i · F̂_a + Σ_i (1−σ_i) · F̂_{r,i}        （F̂ 表示单位向量）
    normalize=False（= 你 my_run.py 当前状态）：
        F = ∏σ_i · F_a + Σ_i (1−σ_i) · F_{r,i}         （原尺度，|F_a|≈d_e³, |F_r|≈d_o³）

    【修正说明】为什么必须归一化：
        |F_a| = ‖η_e‖³ 随目标距离走（目标一般 5~15 m），|F_r| = ‖η_o‖³ 只在障碍附近
        才有量。两者不归一化时，近障碍处吸引力仍大 10~40 倍，∏σ 即使降到 0.1 也压不住，
        于是机器狗"最后一刻才甩开"，最小安全裕度 < ρ_b，违反论文式(3)。
    """
    robot_pos = np.asarray(robot_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    # ---- 吸引场（论文式 6/7/8）：p = −η_e ⇒ F_a = ‖η_e‖²(−η_e) ----
    ne = robot_pos - target_pos
    F_a = float(np.dot(ne, ne)) * (-ne)
    if normalize:                                      # ← 你注释掉的 :14-16 这两行
        n = np.linalg.norm(F_a)
        if n > 1e-8:
            F_a = F_a / n

    F_r = np.zeros(2)
    pre_sigma = 1.0                                    # 论文式(12)：吸引项权重是【连乘 ∏】
    for obs in obstacles:
        obs_pos = np.asarray(obs[:2], dtype=float)
        obs_r = float(obs[2])

        ne_obs = robot_pos - obs_pos
        dist_sq = float(np.dot(ne_obs, ne_obs))

        beta_i = obs_r ** 2 - dist_sq                                     # 式(14)
        beta_a = obs_r ** 2 - (rho_b + rho_a + obs_r) ** 2                # 式(15)
        beta_s = obs_r ** 2 - (rho_b + rho_s + obs_r) ** 2                # 式(16)

        delta_beta = beta_s - beta_a
        divide = beta_a + chi * delta_beta

        if beta_i <= beta_a:
            sigma = 1.0
        elif beta_i <= divide:
            sigma = 1.0 - (beta_i - beta_a) ** 2 / (chi * delta_beta ** 2)          # 式(19)
        elif beta_i <= beta_s:
            sigma = (beta_i - beta_s) ** 2 / ((1.0 - chi) * delta_beta ** 2)        # 式(20)
        else:
            sigma = 0.0

        # ---- 斥力场（式 10/11）：ε=1 ⇒ 纯切向 F_r = ‖η‖²·[y, −x] ----
        F_r_i = dist_sq * np.array([ne_obs[1], -ne_obs[0]])
        if normalize:                                  # ← 你注释掉的 :61-63 这三行
            n = np.linalg.norm(F_r_i)
            if n > 1e-8:
                F_r_i = F_r_i / n

        F_r += (1.0 - sigma) * F_r_i
        pre_sigma *= sigma

    F = pre_sigma * F_a + F_r
    if np.linalg.norm(F) < 1e-12:                      # 场退化兜底（与你 gvf 行为一致）
        return float(np.arctan2(ne[1], ne[0]))
    return float(np.arctan2(F[1], F[0]))


def target_position(t):
    """目标圆周运动（与 my_run.py:215 同步）：(10+2sin(0.2t), 10+2cos(0.2t))。"""
    return np.array([10.0 + 2.0 * np.sin(TGT_OMEGA * t),
                     10.0 + 2.0 * np.cos(TGT_OMEGA * t)])


# ===========================================================================================
# 三、复刻 my_run.main() 的循环，并记录数据
# ===========================================================================================
def simulate(dw, steps=STEPS, verbose=False, normalize=True):
    """与 my_run.main() 相同的执行顺序，只是把每步的量记录下来。

    关键细节（都按你原文保留，不要"顺手优化"）：
      · gvf 用的是【上一步末尾更新过的】target_pos（目标更新在循环末尾，my_run.py:215）
      · 滑模面 s 用的也是同一个（旧的）target_pos（my_run.py:175-177）
      · 观测器在算控制量【之前】推进，且输入是上一步的 r_pre / u_pre（my_run.py:167/181）
      · fai_r_dot 做了 ±π 归一化（my_run.py:156-159）
    """
    p = PARAMS
    robot_state = START.copy()
    target_pos = np.array(TARGET0, dtype=float)

    steering_obs_state, omega = R.init_gpio(robot_state[2], p['m'], p['omega_o'])
    approach_obs_state, _ = R.init_gpio(0.0, p['m'], p['omega_o'])       # 你这里给的初值是 0.0

    r_pre, u_pre = 0.0, 0.0
    fai_r = gvf_field(robot_state[:2], target_pos, OBSTACLES, normalize)
    fai_r_pre = fai_r

    keys = ('t', 'x', 'y', 'fai', 'fai_r', 'e_fai', 'u', 'r',
            'dw3_hat', 'dp_hat', 'dist', 'clear', 'tgt_x', 'tgt_y')
    rec = {k: np.zeros(steps) for k in keys}

    for t in range(steps):
        x, y, fai = robot_state

        # ---- 向量场（my_run.py:154）----
        fai_r = gvf_field(robot_state[:2], target_pos, OBSTACLES, normalize)

        # ---- 参考航向角速度：±π 归一化（my_run.py:156-161）----
        diff = fai_r - fai_r_pre
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        fai_r_dot = diff / DT
        fai_r_pre = fai_r

        # ---- IMU 角速度测量（my_run.py:164）----
        fai_dot_meas = r_pre + dw[2]

        # ---- 转向观测器推进（my_run.py:167，输入是上一步的 r_pre）----
        dw3_hat, steering_obs_state = R.gpio(r_pre, fai, DT, steering_obs_state,
                                             omega, p['m'])

        # ---- 转向控制律（my_run.py:170-172）----
        e_fai = fai - fai_r
        r = -p['k1'] * e_fai - p['k2'] * (fai_dot_meas - fai_r_dot) - dw3_hat
        r = float(np.clip(r, -2.0, 2.0))

        # ---- 滑模面（my_run.py:175-177）----
        s = (x - target_pos[0]) + p['lambda_val'] * (y - target_pos[1])

        # ---- 趋近观测器推进（my_run.py:180-181，输入是上一步的 u_pre）----
        input_u = (np.cos(fai) + p['lambda_val'] * np.sin(fai)) * u_pre
        d_p_hat, approach_obs_state = R.gpio(input_u, s, DT, approach_obs_state,
                                             omega, p['m'])

        # ---- 趋近控制律（my_run.py:184-188）----
        den = np.cos(fai) + p['lambda_val'] * np.sin(fai)
        if abs(den) < 0.12:
            den = 0.12 if den > 0 else -0.12
        u_raw = (-p['k3'] * s - p['k4'] * np.tanh(s / p['epsilon']) - d_p_hat) / den

        # ---- 转向优先（my_run.py:191-194）----
        u = np.exp(p['fai_T'] - abs(e_fai)) * u_raw if abs(e_fai) > p['fai_T'] else u_raw

        # ---- 仅前进 + 限速（my_run.py:198）----
        u = float(np.clip(u, 0.0, p['u_max']))

        # ---- 记录（用"本步执行前"的状态，与你日志的打印口径一致）----
        dist = float(np.hypot(x - target_pos[0], y - target_pos[1]))
        clear = min(float(np.hypot(x - o[0], y - o[1]) - o[2]) for o in OBSTACLES)
        for k, v in (('t', t * DT), ('x', x), ('y', y), ('fai', fai), ('fai_r', fai_r),
                     ('e_fai', e_fai), ('u', u), ('r', r), ('dw3_hat', dw3_hat),
                     ('dp_hat', d_p_hat), ('dist', dist), ('clear', clear),
                     ('tgt_x', target_pos[0]), ('tgt_y', target_pos[1])):
            rec[k][t] = v

        # ---- 积分运动学模型（my_run.py:201-208）----
        robot_state[0] += (u * np.cos(fai) + dw[0]) * DT
        robot_state[1] += (u * np.sin(fai) + dw[1]) * DT
        robot_state[2] += (r + dw[2]) * DT

        # ---- 更新上一时刻（my_run.py:212）----
        r_pre, u_pre = r, u

        # ---- 目标圆周运动（my_run.py:215，更新发生在循环末尾）----
        target_pos = target_position(t * DT)

        if verbose and t % 5000 == 0:
            print(f"  [render] t={t*DT:5.1f}s  位置=({x:6.2f},{y:6.2f})  "
                  f"距目标 {dist:5.2f} m")

    rec['final_state'] = robot_state
    return rec


# ===========================================================================================
# 四、绘图元件
# ===========================================================================================
def _rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# 顶视四足机器狗的局部坐标形状（单位 m，与 ρ_b = 1 m 的安全圆同尺度）
_DOG_BODY = _rect(-0.32, 0.30, -0.16, 0.16)
_DOG_HEAD = _rect(0.30, 0.48, -0.10, 0.10)
_DOG_SNOUT = _rect(0.48, 0.56, -0.05, 0.05)
_DOG_TAIL = _rect(-0.42, -0.32, -0.04, 0.04)
_DOG_LEGS = [
    _rect(0.05, 0.24, 0.16, 0.34),     # 前左
    _rect(0.05, 0.24, -0.34, -0.16),   # 前右
    _rect(-0.26, -0.07, 0.16, 0.34),   # 后左
    _rect(-0.26, -0.07, -0.34, -0.16), # 后右
]


def draw_dog(ax, x, y, psi, scale=1.0, alpha=1.0, lw=0.7, zorder=7):
    """在 (x, y) 处按朝向 psi 画一只"顶视机器狗"。"""
    c, s = np.cos(psi), np.sin(psi)
    Rm = np.array([[c, -s], [s, c]])

    for leg in _DOG_LEGS:
        ax.add_patch(Polygon((leg * scale) @ Rm.T + np.array([x, y]), closed=True,
                             fc="#263238", ec="#111111", lw=lw, alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon((_DOG_TAIL * scale) @ Rm.T + np.array([x, y]), closed=True,
                         fc="#455A64", ec="#111111", lw=lw, alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon((_DOG_BODY * scale) @ Rm.T + np.array([x, y]), closed=True,
                         fc="#37474F", ec="#111111", lw=lw, alpha=alpha, zorder=zorder + 1))
    ax.add_patch(Polygon((_DOG_HEAD * scale) @ Rm.T + np.array([x, y]), closed=True,
                         fc="#455A64", ec="#111111", lw=lw, alpha=alpha, zorder=zorder + 1))
    ax.add_patch(Polygon((_DOG_SNOUT * scale) @ Rm.T + np.array([x, y]), closed=True,
                         fc="#FFC107", ec="#111111", lw=lw, alpha=alpha, zorder=zorder + 2))


def _spread_indices(x, y, n, min_dist):
    """在轨迹上挑 n 个"空间上分散"的时刻（避免快照叠成一团）。"""
    cand = np.linspace(0, len(x) - 1, max(n * 6, n)).astype(int)
    keep: list[int] = []
    for k in cand:
        if all(np.hypot(x[k] - x[j], y[k] - y[j]) >= min_dist for j in keep):
            keep.append(int(k))
        if len(keep) >= n:
            break
    return keep or [0]


def draw_obstacles(ax):
    """障碍物（灰实心圆）+ 安全圆（红虚线，ρ_b+ρ_o）+ 激活圆（灰点线，ρ_b+ρ_o+ρ_a）。"""
    for ob in OBSTACLES:
        x, y, r = ob[0], ob[1], ob[2]
        ax.add_patch(Circle((x, y), r, fc="#c9c9c9", ec="#555555", lw=1.2, zorder=3))
        ax.add_patch(Circle((x, y), r + RHO_B, fc="none", ec="#d62728", ls="--",
                            lw=1.2, alpha=0.85, zorder=3))
        ax.add_patch(Circle((x, y), r + RHO_B + RHO_A, fc="#4a90d9", alpha=0.045,
                            ec="#4a90d9", ls=":", lw=0.9, zorder=2))


def panel(ax, rec, dw, title, n_snapshot=6):
    """画一个"场景 + 轨迹"面板。"""
    t = rec['t']
    x, y, tgt_x, tgt_y = rec['x'], rec['y'], rec['tgt_x'], rec['tgt_y']

    allx = np.concatenate([x, tgt_x, [o[0] for o in OBSTACLES]])
    ally = np.concatenate([y, tgt_y, [o[1] for o in OBSTACLES]])
    pad = 2.0
    ax.set_xlim(allx.min() - pad, allx.max() + pad)
    ax.set_ylim(ally.min() - pad, ally.max() + pad)
    ax.set_aspect("equal", adjustable="box")

    draw_obstacles(ax)

    ax.plot(tgt_x, tgt_y, ls="--", color="#FFB300", lw=1.6, alpha=0.95, zorder=4,
            label="目标轨迹")
    ax.plot(tgt_x[-1], tgt_y[-1], "*", ms=19, color="#FFB300", mec="#7a5200", mew=1.0,
            zorder=9, label="目标（末时刻）")

    ax.plot(x, y, "-", color="#1f77b4", lw=2.0, alpha=0.95, zorder=6,
            label="机器狗轨迹")
    ax.plot(x[0], y[0], "s", ms=8, color="#0d47a1", mec="k", zorder=9, label="起点")

    idx = _spread_indices(x, y, n_snapshot, min_dist=3.0)
    for i, k in enumerate(idx):
        ax.plot(x[k], y[k], "o", ms=4, color="#1f77b4", alpha=0.6, zorder=6)
        draw_dog(ax, x[k], y[k], rec['fai'][k], scale=1.35, alpha=0.38, zorder=7)
        if i == 0:
            ax.annotate(f"t={t[k]:.0f}s", (x[k], y[k]), xytext=(9, -14),
                        textcoords="offset points", fontsize=8, color="#37474F")
    draw_dog(ax, x[-1], y[-1], rec['fai'][-1], scale=1.35, alpha=1.0, zorder=10)
    ax.annotate(f"t={t[-1]:.0f}s", (x[-1], y[-1]), xytext=(10, 6),
                textcoords="offset points", fontsize=8.5, color="#0d47a1",
                fontweight="bold", zorder=12)

    clear_min = float(rec['clear'].min())
    stall = float(np.mean(rec['u'] > 0.05) * 100.0)
    info = (f"时长 {t[-1]:.0f}s   末端距目标 {rec['dist'][-1]:.2f} m\n"
            f"最小安全裕度 {clear_min:.2f} m（判据 > {RHO_B:.1f}）\n"
            f"u>0.05 占比 {stall:.0f}%   平均 u {rec['u'].mean():.2f} m/s")
    ax.text(0.015, 0.985, info, transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9, pad=3.5), zorder=11)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)


def figure_trajectory(cases, out_path):
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(7.4 * n, 7.0), squeeze=False)
    for ax, (title, rec, dw) in zip(axes[0], cases):
        panel(ax, rec, dw, title)
    fig.suptitle("论文复现：机器狗 / 障碍物 / 目标 与双方轨迹（my_run.py 的算法 + 归一化）",
                 fontsize=13.5, y=0.995)
    fig.text(0.5, -0.005,
             "灰实心圆=障碍物  红虚线=安全圆(ρ_b+ρ_o)  蓝色点线区=激活范围(ρ_b+ρ_o+ρ_a)  "
             "★=目标  蓝实线=机器狗轨迹  半透明小狗=沿途快照  黄鼻=朝向",
             ha="center", fontsize=9, color="#555555")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure_zoom(cases, out_path, t_from=22.0):
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 6.2), squeeze=False)
    for ax, (title, rec, dw) in zip(axes[0], cases):
        m = rec['t'] >= t_from
        x, y, tx, ty = rec['x'][m], rec['y'][m], rec['tgt_x'][m], rec['tgt_y'][m]
        tt, ff = rec['t'][m], rec['fai'][m]

        allx = np.concatenate([x, tx]); ally = np.concatenate([y, ty])
        pad = 1.8
        ax.set_xlim(allx.min() - pad, allx.max() + pad)
        ax.set_ylim(ally.min() - pad, ally.max() + pad)
        ax.set_aspect("equal", adjustable="box")

        for ob in OBSTACLES:
            if (ax.get_xlim()[0] - 2 < ob[0] < ax.get_xlim()[1] + 2 and
                    ax.get_ylim()[0] - 2 < ob[1] < ax.get_ylim()[1] + 2):
                ax.add_patch(Circle((ob[0], ob[1]), ob[2], fc="#c9c9c9", ec="#555555", zorder=3))
                ax.add_patch(Circle((ob[0], ob[1]), ob[2] + RHO_B, fc="none", ec="#d62728",
                                    ls="--", lw=1.0, alpha=0.8, zorder=3))

        ax.plot(tx, ty, "--", color="#FFB300", lw=1.5, label="目标轨迹（末段）")
        ax.plot(x, y, "-", color="#1f77b4", lw=2.0, label="机器狗轨迹（末段）")
        ax.plot(x[0], y[0], "s", ms=7, color="#0d47a1", mec="k", label=f"t={t_from:.0f}s 处")

        for k in _spread_indices(x, y, 4, min_dist=1.4):
            draw_dog(ax, x[k], y[k], ff[k], scale=1.25, alpha=0.42, zorder=7)
            ax.annotate(f"{tt[k]:.0f}s", (x[k], y[k]), xytext=(8, -13),
                        textcoords="offset points", fontsize=7.8, color="#37474F")
        draw_dog(ax, x[-1], y[-1], ff[-1], scale=1.35, alpha=1.0, zorder=10)
        ax.plot(tx[-1], ty[-1], "*", ms=18, color="#FFB300", mec="#7a5200", zorder=9)

        ax.text(0.015, 0.985,
                f"末端距目标 {rec['dist'][-1]:.2f} m\n"
                f"min|Δ| ≈ {np.min(np.hypot(x - tx, y - ty)):.2f} m",
                transform=ax.transAxes, va="top", fontsize=8.2,
                bbox=dict(fc="white", ec="#cccccc", alpha=0.9, pad=3.5), zorder=11)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"{title}：末段放大（t>{t_from:.0f}s）", fontsize=10.5)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("末段放大：机器狗与运动目标的相对运动", fontsize=12.5, y=0.99)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure_signals(cases, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4))
    styles = [("-", "#1f77b4"), ("--", "#d62728"), ("-.", "#2ca02c")]

    for i, (label, rec, dw) in enumerate(cases):
        ls, color = styles[i % len(styles)]
        t = rec['t']

        ax = axes[0, 0]
        ax.plot(t, rec['dist'], ls=ls, color=color, lw=1.5, label=label)
        ax.set_ylabel("到目标的距离 [m]")
        ax.set_title("跟踪误差", fontsize=10.5)

        ax = axes[0, 1]
        ax.plot(t, rec['dw3_hat'], ls=ls, color=color, lw=1.4, label=label)
        ax.axhline(dw[2], color=color, ls=":", lw=1.1, alpha=0.8)
        ax.set_ylabel(r"$\hat d_{w3}$  [rad/s]")
        ax.set_title("偏航扰动估计（点线=该组的真值）", fontsize=10.5)

        ax = axes[1, 0]
        ax.plot(t, rec['dp_hat'], ls=ls, color=color, lw=1.4, label=label)
        ax.set_ylabel(r"$\hat d_p$  [m/s]")
        ax.set_xlabel("t [s]")
        ax.set_title("趋近通道扰动估计", fontsize=10.5)

        ax = axes[1, 1]
        ax.plot(t, rec['u'], ls=ls, color=color, lw=1.1, label=label)
        ax.set_ylabel("u [m/s]")
        ax.set_xlabel("t [s]")
        ax.set_ylim(-0.05, PARAMS['u_max'] * 1.1)
        ax.set_title("纵向速度指令（u=0 表示停住）", fontsize=10.5)

    for ax in axes.ravel():
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)
    fig.suptitle("my_run.py 运行结果的关键信号（含归一化）", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ===========================================================================================
# 五、一致性自检：本脚本复刻的循环 vs my_run.main() 的打印坐标
# ===========================================================================================
def check_against_myrun():
    """运行 my_run.main()，把打印出来的位置与 simulate() 的记录对比。

    口径说明：my_run.py 里两处归一化现在是**启用**的（第 13-16、60-63 行），
    所以这里必须用 normalize=True（默认）才能对上；若你以后把归一化注释回去，
    请改用 --check-no-normalize 或把这里改成 normalize=False。
    """
    print("\n[一致性自检] 正在运行 my_run.main()（约 20 s）并对比坐标 ...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.main()
    lines = buf.getvalue().splitlines()

    rec = simulate(DW, normalize=True)
    n_steps = len(rec['t'])
    pat = re.compile(r"Step\s+(\d+)\s+\|\s+位置:\s*\(\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\)")
    worst = 0.0
    n_checked = 0
    for ln in lines:
        m = pat.search(ln)
        if not m:
            continue
        k = int(m.group(1))
        if k >= n_steps:
            continue
        px, py = float(m.group(2)), float(m.group(3))
        worst = max(worst, float(np.hypot(px - rec['x'][k], py - rec['y'][k])))
        n_checked += 1
    if n_checked == 0:
        print("  [warn] 没解析到 my_run 的日志行，跳过对比")
        return None
    tol = 1e-2
    ok = worst < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] 对比 {n_checked} 个时刻的坐标，"
          f"最大偏差 {worst*1000:.2f} mm（判据 < {tol*1000:.0f} mm，受你日志两位小数限制）")
    return worst


# ===========================================================================================
def main():
    ap = argparse.ArgumentParser(description="把 my_run.py 的运行结果画成论文风格的图")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--check", action="store_true", help="与 my_run.main() 的打印结果对比坐标")
    ap.add_argument("--no-nodist", action="store_true", help="只画有扰动的那一组")
    ap.add_argument("--no-normalize", action="store_true", help="关掉归一化（= 你 my_run 当前状态）")
    args = ap.parse_args()

    normalize = not args.no_normalize

    print("=" * 100)
    print("my_render：用 my_run.py 的算法跑出数据并绘图")
    print(f"  参数：dt={DT}s, steps={args.steps}（时长 {args.steps*DT:.0f}s）")
    print(f"  目标：圆心 (10,10) 半径 2 m，ω={TGT_OMEGA} rad/s ⇒ 切向速度 "
          f"{TGT_OMEGA*2:.2f} m/s")
    print(f"  障碍：{len(OBSTACLES)} 个，ρ_b={RHO_B}, ρ_a={RHO_A}, ρ_s={RHO_S}, χ={CHI}")
    print(f"  向量场归一化：{'已启用 ✓（与 my_run.py 第 13-16/60-63 行一致）' if normalize else '未启用（回退到论文原始量纲）'}")
    print("=" * 100)

    cases = []
    print("\n[1] 有扰动 dw = [-0.1, 0.05, 0.4]（你 my_run.py 的默认设置）...")
    rec1 = simulate(DW, steps=args.steps, normalize=normalize)
    cases.append(("有扰动  d_w=(-0.1, 0.05, 0.4)", rec1, DW))

    if not args.no_nodist:
        print("[2] 无扰动 dw = [0, 0, 0]（对照）...")
        rec0 = simulate(np.zeros(3), steps=args.steps, normalize=normalize)
        cases.insert(0, ("无扰动  d_w=(0, 0, 0)", rec0, np.zeros(3)))

    for label, rec, _ in cases:
        print(f"    {label}: 末端距目标 {rec['dist'][-1]:6.2f} m | "
              f"最小安全裕度 {rec['clear'].min():6.2f} m | "
              f"u>0.05 占比 {np.mean(rec['u']>0.05)*100:5.1f}%")

    p1 = figure_trajectory(cases, os.path.join(OUT_DIR, "fig1_trajectory.png"))
    p2 = figure_signals(cases, os.path.join(OUT_DIR, "fig2_signals.png"))
    p3 = figure_zoom(cases, os.path.join(OUT_DIR, "fig3_final_approach.png"))
    print(f"\n[图1] {p1}\n[图2] {p2}\n[图3] {p3}")

    if args.check:
        check_against_myrun()


if __name__ == "__main__":
    main()
