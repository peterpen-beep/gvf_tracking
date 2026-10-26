"""改进制导向量场（Modified Guidance Vector Field）。

对应论文第 III 节：
  式 (4)      基础向量场 F(η) = ε(p^Tη)η − p(η^Tη)
  式 (5)      制导角 ψ_r = arctan(F_y / F_x)   ← 实现用 atan2
  式 (6)(7)   吸引场（ε=2，η_e = η_b − η_t）
  式 (8)      可变属性向量 p = −η_e            ← 改进 1
  式 (9)      Theorem 1 的指向性证明
  式 (10)(11) 斥力场（ε=1，η_e,oi = η_b − η_o,i）
  式 (12)     合成场
  式 (13)     融合权重 σ_i(β_i)
  式 (14)(15)(16) β_i, β_a,i, β_s,i
  式 (17)-(20) 可调融合函数 ξ(β)               ← 改进 2

几何性质（本实现依赖的两个关键事实，见 README）：
  P1. ε=2 且 p=−η_e 时 F_a = ‖η_e‖²·(−η_e)，方向严格指向目标（Theorem 1）。
  P2. ε=1 时恒有 F_r·η = 0，即斥力场处处与径向正交 → 积分曲线是绕障碍物的圆。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .config import GVFConfig


# ----------------------------------------------------------------------------
# 单个场的构造
# ----------------------------------------------------------------------------
def attractive_field(eta_b: np.ndarray, eta_t: np.ndarray, cfg: GVFConfig) -> np.ndarray:
    """吸引场（论文式 (6)(7)）。

    eta_b: 机器人位置 [x, y]；eta_t: 目标位置 [x, y]。
    """
    e = np.asarray(eta_b, dtype=float) - np.asarray(eta_t, dtype=float)
    x, y = e[0], e[1]

    if cfg.att_mode == "variable":
        # 式 (8)：可变属性向量 p = −η_e（改进 1）
        px, py = -x, -y
    else:
        # 经典 GVF：固定属性向量 p（用于图 3 的对照实验）
        px, py = cfg.fixed_p

    # 式 (6)(7)
    fa_x = px * x * x - px * y * y + 2.0 * py * x * y
    fa_y = py * y * y - py * x * x + 2.0 * px * x * y
    return np.array([fa_x, fa_y], dtype=float)


def repulsive_field(eta_b: np.ndarray, eta_o: np.ndarray, sense: float) -> np.ndarray:
    """斥力场（论文式 (10)(11)）。

    【复现缺口 #1】论文未给出斥力场的属性向量 p。本实现取
        p_i = sense · [−y_e,oi, x_e,oi]^T      （缺口#1 的构造 A）
    代入式 (10)(11) 后得到纯切向场
        F_r,i = sense · ‖η_e,oi‖² · [y_e,oi, −x_e,oi]^T
    它满足 P2（F_r·η ≡ 0），积分曲线是以障碍物为圆心的圆；sense = +1 为顺时针。
    """
    e = np.asarray(eta_b, dtype=float) - np.asarray(eta_o, dtype=float)
    x, y = e[0], e[1]
    px, py = -y, x                      # 构造 A 的属性向量
    fr_x = py * x * y - px * y * y      # 式 (10)
    fr_y = px * x * y - py * x * x      # 式 (11)
    return sense * np.array([fr_x, fr_y], dtype=float)


# ----------------------------------------------------------------------------
# β 与融合权重（论文式 (13)-(20)）
# ----------------------------------------------------------------------------
def beta_triplet(d: float, rho_o: float, cfg: GVFConfig) -> Tuple[float, float, float]:
    """返回 (β_i, β_a,i, β_s,i)。d 为机器人到障碍物中心的距离。"""
    beta = rho_o ** 2 - d ** 2                                                  # 式 (14)
    beta_a = rho_o ** 2 - (cfg.rho_b + rho_o + cfg.rho_a) ** 2                  # 式 (15)
    beta_s = rho_o ** 2 - (cfg.rho_b + rho_o + cfg.rho_s) ** 2                  # 式 (16)
    return beta, beta_a, beta_s


def blending_function(beta: float, beta_a: float, beta_s: float, chi: float) -> float:
    """可调融合函数 ξ(β)（论文式 (19)(20)；导数为式 (17)(18)）。

    性质（已数值验证，见 tests/test_verification.py）：
      ξ(β_a) = 1, ξ(β_s) = 0, ξ'(β_a) = ξ'(β_s) = 0，
      在接缝 β_j = β_a + χ(β_s−β_a) 处取值 1−χ 且一阶导连续（C¹）。
    注意：分母是 χ·Δ² 与 (1−χ)·Δ²（Δ = β_s−β_a），不是 (χΔ)²。
    """
    if beta <= beta_a:
        return 1.0
    if beta >= beta_s:
        return 0.0
    delta = beta_s - beta_a
    joint = beta_a + chi * delta
    if beta <= joint:
        return 1.0 - (beta - beta_a) ** 2 / (chi * delta * delta)      # 式 (19)
    return (beta - beta_s) ** 2 / ((1.0 - chi) * delta * delta)        # 式 (20)


def blend_weight(beta: float, beta_a: float, beta_s: float, chi: float) -> float:
    """融合权重 σ_i(β_i)（论文式 (13)）：远（β<β_a）为 1（纯吸引），近（β>β_s）为 0（纯斥力）。"""
    return blending_function(beta, beta_a, beta_s, chi)


# ----------------------------------------------------------------------------
# 绕行方向（缺口 #1 的一部分）
# ----------------------------------------------------------------------------
def circulation_sense(eta_b: np.ndarray, eta_t: np.ndarray, eta_o: np.ndarray,
                      rule: str = "side", default: float = 1.0) -> float:
    """决定绕障碍物的旋向，使机器人从"更合理"的一侧通过。

    side  : 保持机器人当前所在的方位（不跨越"障碍物→目标"这条线）
            sense = −sign( cross(η_b−η_o, η_t−η_o) )
    progress: 让切向速度在"机器人→目标"方向上的投影为正（朝目标推进）
            sense = sign( (η_t−η_b)·[y_e, −x_e] )
    """
    v1 = np.asarray(eta_b, dtype=float) - np.asarray(eta_o, dtype=float)
    v2 = np.asarray(eta_t, dtype=float) - np.asarray(eta_o, dtype=float)
    if rule == "progress":
        tang = np.array([v1[1], -v1[0]])
        proj = float(np.dot(np.asarray(eta_t, dtype=float) - np.asarray(eta_b, dtype=float), tang))
        return float(np.sign(proj)) if abs(proj) > 1e-9 else default
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    if abs(cross) < 1e-9:
        return default
    return -float(np.sign(cross))


# ----------------------------------------------------------------------------
# 合成场（论文式 (12)）
# ----------------------------------------------------------------------------
@dataclass
class FieldDiagnostics:
    """单步向量场的诊断信息（用于绘图与调试）。"""

    psi_r: float
    magnitude: float
    sigmas: np.ndarray          # 每个障碍物的 σ_i
    betas: np.ndarray
    distances: np.ndarray       # 到各障碍物中心的距离
    active: bool                # 是否有障碍物进入激活区


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def resultant_field(eta_b: np.ndarray,
                    eta_t: np.ndarray,
                    obstacle_pos: Sequence[np.ndarray],
                    obstacle_rho: Sequence[float],
                    cfg: GVFConfig,
                    senses: Sequence[float] | None = None,
                    psi_r_prev: float | None = None,
                    ) -> Tuple[np.ndarray, FieldDiagnostics]:
    """合成制导向量场 F(η)（论文式 (12)）并给出制导角 ψ_r。

    论文式 (12) 的原文形式（**吸引项是连乘 ∏，斥力项是求和 ∑**）：

        F(η) = ∏_i σ_i(β_i) · F_a(η_e)  +  Σ_i (1 − σ_i(β_i)) · F_r(η_e,oi)

    语义（很重要）：障碍物**不会**对机器人产生吸引力，它只是"削弱/切断"目标的吸引力。
    连乘的物理含义是：**只要有任意一个障碍物进入 σ_i = 0 的区域，整个吸引项立刻归零**，
    机器人只剩斥力兜底 → 从结构上杜绝了"吸引力把机器人拖进障碍物"的情形。
    `blend` 参数提供三种读法：
        "product"（默认）= ∏_i σ_i        ← 论文原文
        "min"           = min_i σ_i      ← 与连乘同样具备"任一为 0 则整体为 0"的性质
        "sum"           = Σ_i σ_i        ← 【勘误】本实现早期把 ∏ 误读为 Σ 的写法，
                                            保留仅供消融对照（多障碍时会穿入障碍）

    另一处工程化处理（README 缺口#1）：
        normalize_fields=True 时把 F_a 与各 F_r,i 单位化后再融合。
        原因：论文原始标度下 |F_a| ≈ d_target³、|F_r| ≈ d_obstacle³，
        在 d_target ≈ 20 m、d_obstacle ≈ 3 m 时斥力被削弱约 300 倍而失效；
        而斥力场的属性向量标度在论文中未给出。
    """
    eta_b = np.asarray(eta_b, dtype=float)
    eta_t = np.asarray(eta_t, dtype=float)

    fa = attractive_field(eta_b, eta_t, cfg)
    if cfg.normalize_fields:
        fa_dir = _unit(fa)
    else:
        fa_dir = fa

    sigmas: List[float] = []
    betas: List[float] = []
    dists: List[float] = []
    fr_sum = np.zeros(2, dtype=float)
    fr_dirs: List[np.ndarray] = []

    for i, (eta_o, rho_o) in enumerate(zip(obstacle_pos, obstacle_rho)):
        eta_o = np.asarray(eta_o, dtype=float)
        d = float(np.linalg.norm(eta_b - eta_o))
        beta, beta_a, beta_s = beta_triplet(d, float(rho_o), cfg)
        sigma = blend_weight(beta, beta_a, beta_s, cfg.chi)
        sigmas.append(sigma)
        betas.append(beta)
        dists.append(d)

        if senses is not None and len(senses) > i:
            sense = float(senses[i])
        else:
            sense = circulation_sense(eta_b, eta_t, eta_o, cfg.rep_sense_rule)
        fr = repulsive_field(eta_b, eta_o, sense) * cfg.rep_strength
        if cfg.normalize_fields:
            fr = _unit(fr)
        fr_dirs.append(fr)
        fr_sum = fr_sum + (1.0 - sigma) * fr

    sig = np.asarray(sigmas, dtype=float) if sigmas else np.zeros(0)

    # ---- 吸引项权重（论文式 (12) 的前半部分）----
    if not len(sig):                     # 无障碍物 → 退化为纯吸引场
        w_att = 1.0
    elif cfg.blend == "product":
        w_att = float(np.prod(sig))      # ★ 论文原文：∏_i σ_i（连乘）
    elif cfg.blend == "min":
        w_att = float(np.min(sig))       # 等价稳健写法：任一 σ_i = 0 ⇒ 整体为 0
    elif cfg.blend == "sum":
        w_att = float(np.sum(sig))       # 【勘误】早期误读为 Σσ_i，仅供消融对照
    else:
        raise ValueError(f"未知的 blend 读法：{cfg.blend!r}（可选 product / min / sum）")

    F = w_att * fa_dir + fr_sum

    mag = float(np.linalg.norm(F))
    if mag < cfg.result_eps:
        # 合成场退化（吸引与斥力恰好抵消）时保持上一时刻制导角，避免 ψ_r 跳变
        if psi_r_prev is not None:
            psi_r = float(psi_r_prev)
        else:
            psi_r = float(np.arctan2(eta_t[1] - eta_b[1], eta_t[0] - eta_b[0]))
        F = np.array([np.cos(psi_r), np.sin(psi_r)], dtype=float)
        mag = 1.0
    else:
        psi_r = float(np.arctan2(F[1], F[0]))           # 式 (5)

    diag = FieldDiagnostics(
        psi_r=psi_r,
        magnitude=mag,
        sigmas=sig,
        betas=np.asarray(betas, dtype=float),
        distances=np.asarray(dists, dtype=float),
        active=bool(len(sig) and np.any(sig < 1.0 - 1e-12)),
    )
    return F, diag


def guidance_angle(eta_b: np.ndarray, eta_t: np.ndarray,
                   obstacle_pos: Sequence[np.ndarray] = (),
                   obstacle_rho: Sequence[float] = (),
                   cfg: GVFConfig | None = None,
                   psi_r_prev: float | None = None) -> float:
    """便捷函数：直接返回制导角 ψ_r。"""
    cfg = cfg or GVFConfig()
    _, diag = resultant_field(eta_b, eta_t, obstacle_pos, obstacle_rho, cfg,
                              psi_r_prev=psi_r_prev)
    return diag.psi_r


# ----------------------------------------------------------------------------
# 工具：积分曲线（用于图 3 的可视化）
# ----------------------------------------------------------------------------
def integrate_field_line(eta0: np.ndarray,
                         target: np.ndarray,
                         cfg: GVFConfig,
                         steps: int = 4000,
                         ds: float = 0.01,
                         stop_radius: float = 0.05) -> np.ndarray:
    """沿单位化向量场积分出一条积分曲线（Euler，足够用于可视化）。"""
    pts = [np.asarray(eta0, dtype=float)]
    p = pts[0].copy()
    for _ in range(steps):
        F, _ = resultant_field(p, target, (), (), cfg)
        n = float(np.linalg.norm(F))
        if n < 1e-12:
            break
        p = p + ds * F / n
        pts.append(p.copy())
        if float(np.linalg.norm(p - target)) < stop_radius:
            break
        if np.any(np.abs(p) > 1e4):
            break
    return np.asarray(pts)
