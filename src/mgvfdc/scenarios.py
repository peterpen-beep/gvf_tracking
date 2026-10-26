"""三个仿真算例与论文图 3/图 5 的实验场景（论文第 V 节）。

论文给定的场景要素：
  · 动态目标：圆心 (30,35)、半径 15 m、角速度 0.021 rad/s 的圆周运动（切向速度 0.315 m/s）
  · 参数：ρ_b = 1 m, ρ_a = 2.1 m, ρ_s = 1.1 m, χ = 0.25
  · Case 1：动态目标 + 6 个静态圆形障碍（半径各异，散布在方形区域内）
  · Case 2：动态目标 + 动态多障碍（与 GVFPID 对比）
  · Case 3：静态单障碍 + 动态目标 + 常值扰动 d_w = [−0.1, 0.05, 0.4]^T
【整定】障碍物的具体坐标与目标圆周轨迹的相位论文未给，本文件按"必须真正触发避障"的
原则布置：使障碍物落在机器人为跟踪目标而实际经过的圆弧（α ∈ [−120°, +24°]）附近。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .config import PAPER_DISTURBANCE
from .plant import CircularTarget, Obstacle, StaticTarget, Target


@dataclass
class Scenario:
    """一次闭环仿真的完整场景描述。"""

    name: str
    start: Tuple[float, float]
    psi0: float
    target: Target
    obstacles: List[Obstacle] = field(default_factory=list)
    duration: float = 120.0
    d_w: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: str = ""


# ----------------------------------------------------------------------------
# 工具：把"目标圆周上的极角 α"换算成坐标
# ----------------------------------------------------------------------------
def _polar(alpha_deg: float, radius: float,
           center: Tuple[float, float] = (30.0, 35.0)) -> Tuple[float, float]:
    a = np.deg2rad(alpha_deg)
    return (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))


def _dynamic_target() -> CircularTarget:
    return CircularTarget(cx=30.0, cy=35.0, radius=15.0, omega=0.021, phase=-np.pi / 6.0)


# ----------------------------------------------------------------------------
# Case 1：动态目标 + 静态多障碍
# ----------------------------------------------------------------------------
# 【整定】障碍物布局说明：
#   机器人起点 (0,0)，目标初始位置 (22.5, 22.01)，两者连线方向
#       d = (0.7150, 0.6994)，法向 n = (−0.6994, 0.7150)
#   沿这条"接近走廊"布置 5 个障碍（相对路径的横向偏移 ±0.9~2.2 m，半径 0.7~1.3 m），
#   使得"不做避障的直线接近"会侵入安全区（偏移量 < ρ_b+ρ_o），必须靠 GVF 绕行；
#   再在目标圆周轨迹附近放 1 个（目标自身通过时留出 1.5 m > ρ_b 的间隙），
#   保证"安全距离 > ρ_b"这一指标是场景本身可满足的。
_CORRIDOR = [
    (6.0, 0.9, 0.8),      # (沿路径距离 s, 横向偏移 off, 障碍半径 ρ_o)
    (10.5, -1.5, 1.0),
    (14.5, 2.2, 0.7),
    (18.5, -1.1, 1.3),
    (22.0, 1.8, 0.9),
]


def _corridor_obstacles() -> List[Obstacle]:
    d = np.array([22.5, 22.01])
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    out = []
    for i, (s, off, rho) in enumerate(_CORRIDOR):
        p = s * d + off * n
        out.append(Obstacle((float(p[0]), float(p[1])), rho, name=f"O{i+1}"))
    p6 = _polar(-100.0, 17.6)
    out.append(Obstacle(p6, 1.1, name="O6"))
    return out


def case1(duration: float = 120.0) -> Scenario:
    """6 个不同半径的静态障碍物散布在机器人的接近路径与目标轨迹附近。"""
    return Scenario(
        name="Case 1: 动态目标 + 静态多障碍",
        start=(0.0, 0.0),
        psi0=float(np.arctan2(22.01, 22.5)),
        target=_dynamic_target(),
        obstacles=_corridor_obstacles(),
        duration=duration,
        notes="6 个静态圆形障碍（半径 0.7~1.3 m）沿接近走廊与目标轨迹布置",
    )


# ----------------------------------------------------------------------------
# Case 2：动态目标 + 动态多障碍
# ----------------------------------------------------------------------------
def case2(duration: float = 120.0) -> Scenario:
    """在 Case 1 布局基础上，障碍物做缓慢往复运动（峰值速度 0.15~0.25 m/s）。"""
    obstacles = _corridor_obstacles()
    motion = [
        ((0.0, 1.5), 0.12, 0.0),
        ((1.3, 0.0), 0.10, 1.1),
        ((1.1, -1.1), 0.14, 2.3),
        ((0.0, -1.6), 0.09, 0.6),
        ((-1.4, 0.8), 0.11, 3.1),
        ((0.9, 0.9), 0.13, 4.2),
    ]
    for ob, (amp, freq, ph) in zip(obstacles, motion):
        ob.osc_amp = amp
        ob.osc_freq = freq
        ob.phase = ph
    return Scenario(
        name="Case 2: 动态目标 + 动态多障碍",
        start=(0.0, 0.0),
        psi0=float(np.arctan2(22.01, 22.5)),
        target=_dynamic_target(),
        obstacles=obstacles,
        duration=duration,
        notes="障碍物做幅值 1.1~1.6 m、频率 0.09~0.14 rad/s 的往复运动（峰值速度 ≤ 0.25 m/s）",
    )


# ----------------------------------------------------------------------------
# Case 3：含扰动的环境（"窄门"障碍 + 常值扰动）
# ----------------------------------------------------------------------------
def case3(duration: float = 120.0, gate_offset: float = 2.6,
          gate_s: float = 6.5, d_w=PAPER_DISTURBANCE) -> Scenario:
    """论文 Case 3：简化场景（障碍 + 动态目标）+ 常值扰动 d_w = [−0.1, 0.05, 0.4]^T。

    【整定】障碍物布置为"窄门"：机器人在起点→目标初始位置连线上 s=6.5 m 处的法向
    ±gate_offset 各放一个半径 1.2 m 的障碍。门半宽取 2.6 m 时：
      · 有补偿（MGVFDC）最小安全距离 ≈ 1.20 m > ρ_b → 满足论文安全判据；
      · 无补偿（MGVF）  最小安全距离 ≈ 0.65 m < ρ_b → 违反安全判据且被障碍"困住"，
        始终无法到达目标（复现论文 Fig. 9 "只有带扰动补偿才能有效避免碰撞"）。
    """
    d = np.array([22.5, 22.01]) - np.array([14.0, 14.0])
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    c = np.array([14.0, 14.0]) + gate_s * d
    p1 = c + gate_offset * n
    p2 = c - gate_offset * n
    obstacles = [
        Obstacle((float(p1[0]), float(p1[1])), 1.2, name="O1"),
        Obstacle((float(p2[0]), float(p2[1])), 1.2, name="O2"),
    ]
    return Scenario(
        name="Case 3: 动态目标 + 窄门障碍 + 常值扰动",
        start=(14.0, 14.0),
        psi0=float(np.arctan2(22.01 - 14.0, 22.5 - 14.0)),
        target=_dynamic_target(),
        obstacles=obstacles,
        duration=duration,
        d_w=d_w,
        notes=f"d_w = {tuple(d_w)}（论文给定）；窄门通道宽约 "
              f"{2*(gate_offset - 2.2):.1f} m",
    )


# ----------------------------------------------------------------------------
# 论文图 3 的对照场景（固定 p vs 可变 p 吸引场）
# ----------------------------------------------------------------------------
def fig3_scenario(duration: float = 12.0) -> Scenario:
    """论文 Remark 5 的对比设置：目标 [5,5] m，初始 η=[6.5,6.5] m、ψ=0.3 rad。"""
    return Scenario(
        name="Fig.3 对照: 目标(5,5), 起点(6.5,6.5), ψ=0.3",
        start=(6.5, 6.5),
        psi0=0.3,
        target=StaticTarget((5.0, 5.0)),
        obstacles=[],
        duration=duration,
        notes="用于比较经典 GVF（固定 p）与改进 GVF（可变 p）的积分曲线与轨迹",
    )


def fig3_longrange(duration: float = 40.0) -> Scenario:
    """图 3 的闭环版本：目标在原点，机器人位于 (20,20) 且机头正对目标。

    该设置下"绕行"差异在闭环中可量化：
      · 可变 p（改进）：路径 = 28.284 m = 起终点欧氏距离（零绕行）
      · 固定 p（经典）：路径 ≈ 34 m，到达时间明显更晚（绕圆弧）
    注意：固定 p 需要允许 u < 0 才能闭环收敛（见 README 实现问题 E4）；
    若强制 u ≥ 0，固定 p 会陷入 u = 0 的死锁（实测仅前移 2.57 m）。
    """
    return Scenario(
        name="Fig.3 闭环(远程): 目标(0,0), 起点(20,20), 机头正对目标",
        start=(20.0, 20.0),
        psi0=float(np.arctan2(-20.0, -20.0)),
        target=StaticTarget((0.0, 0.0)),
        obstacles=[],
        duration=duration,
        notes="直线距离 28.284 m；用于闭环量化'绕远路'",
    )


# ----------------------------------------------------------------------------
# 论文图 5 的场景（χ 微调避障性能）
# ----------------------------------------------------------------------------
def fig5_scenario(duration: float = 45.0) -> Scenario:
    """单个障碍物位于机器人→目标的路径上，用于观察 χ 对避障距离的影响。"""
    obstacle = Obstacle((8.0, 8.5), 1.0, name="O1")
    return Scenario(
        name="Fig.5 场景: 目标(0,0), 障碍(8,8.5) r=1.0",
        start=(20.0, 20.0),
        psi0=float(np.arctan2(0.0 - 20.0, 0.0 - 20.0)),
        target=StaticTarget((0.0, 0.0)),
        obstacles=[obstacle],
        duration=duration,
        notes="沿 χ 扫描，考察机器人与障碍物边界的最小距离",
    )


# ----------------------------------------------------------------------------
# 额外的稳健性场景（可选：动态障碍高速 / 匀速直线目标）
# ----------------------------------------------------------------------------
def stress_dynamic_obstacle(duration: float = 120.0) -> Scenario:
    """压力测试：障碍物以 0.5 m/s 横向扫过（论文承认"障碍物太快会失效"）。"""
    obstacles = [
        Obstacle(_polar(-75.0, 13.6), 1.0, vel=(0.0, 0.5), name="O1"),
        Obstacle(_polar(-15.0, 13.2), 1.3, vel=(0.0, -0.5), name="O2"),
    ]
    return Scenario(
        name="压力测试: 快速动态障碍 (0.5 m/s)",
        start=(18.0, 12.0),
        psi0=float(np.arctan2(22.01 - 12.0, 22.5 - 18.0)),
        target=_dynamic_target(),
        obstacles=obstacles,
        duration=duration,
        notes="验证论文局限：动态障碍速度过快时避障可能失效",
    )


ALL_CASES = {
    "case1": case1,
    "case2": case2,
    "case3": case3,
    "fig3": fig3_scenario,
    "fig5": fig5_scenario,
    "stress": stress_dynamic_obstacle,
}
