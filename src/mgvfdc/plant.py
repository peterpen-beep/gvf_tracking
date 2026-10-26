"""被控对象与场景元素：论文式 (1) 的独轮车运动学模型、目标、圆形障碍物。

论文式 (1)：
    η̇ = [[cosψ, −sinψ, 0],
          [sinψ,  cosψ, 0],
          [0,     0,    1]] · [u, v, r]^T + d_w
按 Remark 1（欠驱动，v ≡ 0）退化为：
    ẋ = u cosψ + d_w1,  ẏ = u sinψ + d_w2,  ψ̇ = r + d_w3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# 独轮车运动学（式 (1)）
# ----------------------------------------------------------------------------
@dataclass
class Unicycle:
    """按式 (1) 积分的机器人运动学模型（v ≡ 0）。"""

    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0

    @property
    def eta(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def copy(self) -> "Unicycle":
        return Unicycle(self.x, self.y, self.psi)

    # ------------------------------------------------------------------
    def _f(self, state: np.ndarray, u: float, r: float, d_w: Sequence[float]) -> np.ndarray:
        _, _, psi = state
        return np.array([
            u * np.cos(psi) + d_w[0],
            u * np.sin(psi) + d_w[1],
            r + d_w[2],
        ], dtype=float)

    def step(self, u: float, r: float, d_w: Sequence[float], dt: float) -> None:
        """RK4 积分一步（控制量与扰动在步内保持不变）。"""
        s = np.array([self.x, self.y, self.psi], dtype=float)
        k1 = self._f(s, u, r, d_w)
        k2 = self._f(s + 0.5 * dt * k1, u, r, d_w)
        k3 = self._f(s + 0.5 * dt * k2, u, r, d_w)
        k4 = self._f(s + dt * k3, u, r, d_w)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x, self.y = float(s[0]), float(s[1])
        self.psi = wrap_angle(float(s[2]))


# ----------------------------------------------------------------------------
# 目标（质点，论文将其建模为 mass point）
# ----------------------------------------------------------------------------
class Target:
    """目标基类：提供位置与速度。"""

    def position(self, t: float) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def velocity(self, t: float) -> np.ndarray:
        return np.zeros(2, dtype=float)


class StaticTarget(Target):
    def __init__(self, pos: Sequence[float]):
        self.pos = np.asarray(pos, dtype=float)

    def position(self, t: float) -> np.ndarray:
        return self.pos.copy()


class CircularTarget(Target):
    """论文第 V 节的动态目标：

        x_t(t) = 30 + 15 sin(0.021 t − π/6)
        y_t(t) = 35 − 15 cos(0.021 t − π/6)

    等价于圆心 (30, 35)、半径 15 m、角速度 0.021 rad/s 的圆周运动
    （切向速度 15×0.021 = 0.315 m/s）。
    """

    def __init__(self, cx: float = 30.0, cy: float = 35.0,
                 radius: float = 15.0, omega: float = 0.021,
                 phase: float = -np.pi / 6.0):
        self.cx, self.cy = cx, cy
        self.radius, self.omega, self.phase = radius, omega, phase

    def position(self, t: float) -> np.ndarray:
        phi = self.omega * t + self.phase
        return np.array([self.cx + self.radius * np.sin(phi),
                         self.cy - self.radius * np.cos(phi)], dtype=float)

    def velocity(self, t: float) -> np.ndarray:
        phi = self.omega * t + self.phase
        return np.array([self.radius * self.omega * np.cos(phi),
                         self.radius * self.omega * np.sin(phi)], dtype=float)


class LinearTarget(Target):
    """匀速直线目标（用于补充实验）。"""

    def __init__(self, pos: Sequence[float], vel: Sequence[float]):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)

    def position(self, t: float) -> np.ndarray:
        return self.pos + self.vel * t

    def velocity(self, t: float) -> np.ndarray:
        return self.vel.copy()


# ----------------------------------------------------------------------------
# 圆形障碍物
# ----------------------------------------------------------------------------
@dataclass
class Obstacle:
    """圆形障碍物：p(t) = p0 + v·t + A·sin(ω t + φ)。

    Case 1 用静态（v = 0, A = 0）；Case 2 用动态（v ≠ 0 或 A ≠ 0）。
    """

    pos0: Tuple[float, float]
    radius: float
    vel: Tuple[float, float] = (0.0, 0.0)
    osc_amp: Tuple[float, float] = (0.0, 0.0)
    osc_freq: float = 0.0
    phase: float = 0.0
    name: str = ""

    def position(self, t: float) -> np.ndarray:
        p = np.asarray(self.pos0, dtype=float) + np.asarray(self.vel, dtype=float) * t
        amp = np.asarray(self.osc_amp, dtype=float)
        if np.any(amp != 0.0):
            p = p + amp * np.sin(self.osc_freq * t + self.phase)
        return p

    def velocity(self, t: float) -> np.ndarray:
        v = np.asarray(self.vel, dtype=float).copy()
        amp = np.asarray(self.osc_amp, dtype=float)
        if np.any(amp != 0.0):
            v = v + amp * self.osc_freq * np.cos(self.osc_freq * t + self.phase)
        return v


def obstacle_arrays(obstacles: Sequence[Obstacle], t: float) -> Tuple[list, list]:
    """返回 (位置列表, 半径列表)。"""
    pos = [ob.position(t) for ob in obstacles]
    rho = [ob.radius for ob in obstacles]
    return pos, rho


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def wrap_angle(a: float) -> float:
    """把角度归一化到 (−π, π]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def safety_clearance(eta_b: np.ndarray, obstacles: Sequence[Obstacle],
                     t: float) -> np.ndarray:
    """安全距离指标（论文 Fig. 8）：到各障碍物边界的距离 d_i − ρ_o,i。"""
    out = []
    for ob in obstacles:
        d = float(np.linalg.norm(np.asarray(eta_b, dtype=float) - ob.position(t)))
        out.append(d - ob.radius)
    return np.asarray(out, dtype=float)
