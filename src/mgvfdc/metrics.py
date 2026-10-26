"""性能指标：对应论文第 V 节使用的评价量。

  · 稳态位置跟踪 RMSE（论文 Case 1：x 方向 0.075 m，y 方向 0.091 m）
  · 制导角跟踪误差（论文："almost up to zero"）
  · 安全距离指标（论文 Fig. 8：d_i − ρ_o,i 需始终大于 ρ_b）
  · 路径长度 / 到达时间 / 绕行度（论文 Fig. 7 的"轨迹更曲折、耗时更长"）
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .sim import SimResult


# ----------------------------------------------------------------------------
def rmse(a: np.ndarray, mask: np.ndarray | None = None) -> float:
    """均方根误差。"""
    if mask is not None:
        a = a[mask]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(a))))


def steady_mask(t: np.ndarray, frac: float = 0.30) -> np.ndarray:
    """稳态窗口：默认取仿真时长的最后 30%。"""
    t0 = t[0] + (1.0 - frac) * (t[-1] - t[0])
    return t >= t0


# ----------------------------------------------------------------------------
def path_length(x: np.ndarray, y: np.ndarray) -> float:
    """轨迹弧长 [m]（用于"绕行度"对比）。"""
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))


def arrival_time(t: np.ndarray, dist: np.ndarray, thresh: float = 1.0) -> float:
    """首次进入目标 thresh 邻域的时刻；未到达则返回 nan。"""
    idx = np.where(dist <= thresh)[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def tortuosity(x: np.ndarray, y: np.ndarray) -> float:
    """绕行度 = 路径长度 / 起终点直线距离。"""
    straight = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if straight < 1e-9:
        return float("nan")
    return path_length(x, y) / straight


# ----------------------------------------------------------------------------
def summarize(res: SimResult, steady_frac: float = 0.30) -> Dict[str, float]:
    """汇总一次仿真的全部指标。"""
    m = steady_mask(res.t, steady_frac)
    clr = res.clearance
    n_obs = res.n_obs

    out: Dict[str, float] = {
        "controller": res.controller,
        "scenario": res.scenario,
        "rmse_err_x": rmse(res.err_x),
        "rmse_err_y": rmse(res.err_y),
        "rmse_pos": rmse(np.hypot(res.err_x, res.err_y)),
        "ss_rmse_err_x": rmse(res.err_x, m),
        "ss_rmse_err_y": rmse(res.err_y, m),
        "ss_rmse_pos": rmse(np.hypot(res.err_x, res.err_y), m),
        "rmse_e_psi": rmse(res.e_psi),
        # 制导角跟踪误差：仅在机器人距目标 > 2 m 时统计（贴到目标上时"目标方位角"本身无定义，
        # 会把数值抖动计入误差，论文所说"制导角误差几乎为零"应在此意义下理解）
        "rmse_e_psi_far": rmse(res.e_psi, res.dist > 2.0),
        "ss_rmse_e_psi": rmse(res.e_psi, m),
        "path_length": path_length(res.x, res.y),
        "tortuosity": tortuosity(res.x, res.y),
        "arrival_time_1m": arrival_time(res.t, res.dist, 1.0),
        "arrival_time_3m": arrival_time(res.t, res.dist, 3.0),
        "final_dist": float(res.dist[-1]),
        "mean_u": float(np.mean(res.u)),
        "min_clearance": float(clr.min()) if n_obs else float("inf"),
        "collided": float(clr.min() < 1.0) if n_obs else 0.0,
    }

    if n_obs:
        min_per_obs = res.min_clearance()
        out["min_clearance_worst_obs"] = float(min_per_obs.min())
        out["collision_count"] = float(np.sum(clr.min(axis=0) < 1.0))
    return out


def format_table(rows: list[Dict[str, float]], keys: list[str] | None = None) -> str:
    """把若干指标字典格式化为文本表。"""
    if not rows:
        return ""
    keys = keys or [k for k in rows[0].keys()]
    widths = {k: max(len(k), 12) for k in keys}
    head = "  ".join(k.ljust(widths[k]) for k in keys)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, float("nan"))
            cells.append(f"{v:.4f}".ljust(widths[k]) if isinstance(v, (int, float)) else str(v).ljust(widths[k]))
        lines.append("  ".join(cells))
    return "\n".join(lines)
