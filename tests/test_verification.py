"""论文理论性质的数值验证（可直接运行：python tests/test_verification.py）。

逐条核验论文的解析结论，任何一条失败都意味着复现实现有误：
  T1  Theorem 1：可变属性向量下吸引场方向严格指向目标（夹角为 0）
  T2  改进 1 的几何效果：固定 p 时存在绕远路（积分曲线为曲线），可变 p 时为直线
  T3  斥力场的切向性：F_r · η_e,oi ≡ 0（积分曲线是绕障碍物的圆）
  T4  融合函数 ξ(β)：端点值、端点导数为 0、接缝取值 1−χ、C¹ 连续、单调递减
  T5  β 的单调性语义：σ=1 ⇔ 在激活距离之外；σ=0 ⇔ 在最小安全距离之内
  T6  Theorem 2：GPIO 对常值/多项式型扰动渐近收敛
  T7  Theorem 3：滑模面 s 在有限时间内收敛（含常值扰动）
  T8  运动学模型：无扰动下匀速直线/等角速度圆周的解析一致性
  T9  式 (27) 分母奇异性：奇异保护确实生效（对照实验）
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mgvfdc.config import GVFConfig, GPIOConfig, GuidanceConfig
from mgvfdc.gvf import (attractive_field, beta_triplet, blending_function,
                        circulation_sense, integrate_field_line, repulsive_field,
                        resultant_field)
from mgvfdc.observer import GPIO
from mgvfdc.plant import Unicycle, wrap_angle

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------------
def t1_theorem1() -> None:
    print("\nT1  Theorem 1：吸引场方向严格指向目标（式 (8)(9)）")
    cfg = GVFConfig(att_mode="variable")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        eta_b = rng.uniform(-50, 50, size=2)
        eta_t = rng.uniform(-50, 50, size=2)
        e = eta_b - eta_t
        if np.linalg.norm(e) < 1e-6:
            continue
        F = attractive_field(eta_b, eta_t, cfg)
        to_target = -e                                   # 机器人 → 目标
        cos = float(np.dot(F, to_target) / (np.linalg.norm(F) * np.linalg.norm(to_target)))
        worst = max(worst, abs(1.0 - cos))
    check("F_a 与 −η_e 夹角为 0（cos ≡ 1）", worst < 1e-9, f"max|1−cos| = {worst:.2e}")

    # 幅值应为 ‖η_e‖³（论文推导的直接推论）
    eta_b = np.array([3.0, 4.0])
    eta_t = np.array([0.0, 0.0])
    F = attractive_field(eta_b, eta_t, cfg)
    expected = 5.0 ** 3
    check("‖F_a‖ = ‖η_e‖³", abs(np.linalg.norm(F) - expected) < 1e-9,
          f"{np.linalg.norm(F):.6f} vs {expected:.6f}")


def t2_detour() -> None:
    print("\nT2  改进 1 的几何效果：固定 p 绕远路 vs 可变 p 走直线（图 3）")
    target = np.array([5.0, 5.0])
    start = np.array([6.5, 6.5])

    cfg_fixed = GVFConfig(att_mode="fixed", fixed_p=(1.0, 0.0))
    cfg_var = GVFConfig(att_mode="variable")

    line_fixed = integrate_field_line(start, target, cfg_fixed)
    line_var = integrate_field_line(start, target, cfg_var)

    len_fixed = float(np.sum(np.hypot(*np.diff(line_fixed, axis=0).T)))
    len_var = float(np.sum(np.hypot(*np.diff(line_var, axis=0).T)))
    straight = float(np.linalg.norm(start - target))

    check("固定 p：积分曲线明显绕远（长度 > 1.3× 直线距离）",
          len_fixed > 1.3 * straight, f"{len_fixed:.3f} m vs 直线 {straight:.3f} m")
    check("可变 p：积分曲线近似直线（长度 ≲ 1.02× 直线距离）",
          len_var < 1.02 * straight, f"{len_var:.3f} m")

    # 可变 p 下曲线应严格共线：与 (start−target) 方向的垂直偏差为 0
    d = (start - target) / straight
    perp = np.array([-d[1], d[0]])
    dev = float(np.max(np.abs((line_var - target) @ perp)))
    check("可变 p：整条积分曲线与目标连线共线", dev < 1e-6, f"max 垂直偏差 = {dev:.2e} m")


def t3_tangential() -> None:
    print("\nT3  斥力场切向性：F_r · η_e,oi ≡ 0（积分曲线为圆）")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(2000):
        eta_b = rng.uniform(-20, 20, size=2)
        eta_o = rng.uniform(-20, 20, size=2)
        e = eta_b - eta_o
        if np.linalg.norm(e) < 1e-6:
            continue
        Fr = repulsive_field(eta_b, eta_o, sense=1.0)
        worst = max(worst, abs(float(np.dot(Fr, e))) / (np.linalg.norm(Fr) * np.linalg.norm(e)))
    check("斥力场与径向正交（cos ≡ 0）", worst < 1e-12, f"max|cos| = {worst:.2e}")

    # 沿单位切向场积分（RK4，避免显式欧拉的系统性外扩误差），半径应保持不变（圆）
    def _tangent_unit(state: np.ndarray) -> np.ndarray:
        Fr = repulsive_field(state, np.zeros(2), sense=1.0)
        return Fr / np.linalg.norm(Fr)

    eta_o = np.array([0.0, 0.0])
    p = np.array([3.0, 0.4])
    r0 = float(np.linalg.norm(p))
    ds = 0.001
    for _ in range(20000):
        k1 = _tangent_unit(p)
        k2 = _tangent_unit(p + 0.5 * ds * k1)
        k3 = _tangent_unit(p + 0.5 * ds * k2)
        k4 = _tangent_unit(p + ds * k3)
        p = p + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    drift = abs(float(np.linalg.norm(p)) - r0)
    check("沿斥力场积分（RK4）的半径漂移 < 1e-9 m（确为圆）", drift < 1e-9,
          f"drift = {drift:.2e} m，绕行 {20000*ds:.0f} m")


def t4_blending() -> None:
    print("\nT4  融合函数 ξ(β) 的性质（式 (17)-(20)）")
    for chi in (0.1, 0.25, 0.5, 0.75, 0.9):
        rho_b, rho_a, rho_s, rho_o = 1.0, 2.1, 1.1, 1.0
        cfg = GVFConfig(rho_b=rho_b, rho_a=rho_a, rho_s=rho_s, chi=chi)
        _, beta_a, beta_s = beta_triplet(1.0, rho_o, cfg)
        delta = beta_s - beta_a
        joint = beta_a + chi * delta

        eps = 1e-9
        v_a = blending_function(beta_a, beta_a, beta_s, chi)
        v_s = blending_function(beta_s, beta_a, beta_s, chi)
        v_j = blending_function(joint, beta_a, beta_s, chi)

        # 数值导数
        d_a = (blending_function(beta_a + 1e-6, beta_a, beta_s, chi) - v_a) / 1e-6
        d_s = (v_s - blending_function(beta_s - 1e-6, beta_a, beta_s, chi)) / 1e-6
        d_jm = (v_j - blending_function(joint - 1e-6, beta_a, beta_s, chi)) / 1e-6
        d_jp = (blending_function(joint + 1e-6, beta_a, beta_s, chi) - v_j) / 1e-6

        ok = (abs(v_a - 1.0) < eps and abs(v_s) < eps and abs(v_j - (1.0 - chi)) < 1e-6
              and abs(d_a) < 1e-3 and abs(d_s) < 1e-3 and abs(d_jm - d_jp) < 1e-2)
        check(f"χ={chi}: ξ(β_a)=1, ξ(β_s)=0, ξ(接缝)=1−χ, 端点导数=0, C¹ 连续", ok,
              f"ξ_j={v_j:.4f} (期望 {1-chi:.4f}), Δξ'={abs(d_jm-d_jp):.2e}")

    # 单调递减
    cfg = GVFConfig(chi=0.25)
    _, beta_a, beta_s = beta_triplet(1.0, 1.0, cfg)
    betas = np.linspace(beta_a, beta_s, 500)
    vals = np.array([blending_function(b, beta_a, beta_s, cfg.chi) for b in betas])
    check("ξ 在 [β_a, β_s] 上严格单调递减", bool(np.all(np.diff(vals) < 1e-12)),
          f"最小差分 = {np.min(np.diff(vals)):.2e}")


def t5_sigma_semantics() -> None:
    print("\nT5  σ 的语义：远→纯吸引，近→纯斥力（式 (13)-(16)）")
    cfg = GVFConfig(rho_b=1.0, rho_a=2.1, rho_s=1.1, chi=0.25)
    rho_o = 1.0
    far = 1.0 + 1.0 + 2.1 + 0.5      # 激活距离之外
    mid = 1.0 + 1.0 + 1.6            # 混合区
    near = 1.0 + 1.0 + 1.1 - 0.1     # 最小安全距离之内
    sig = []
    for d in (far, mid, near):
        beta, ba, bs = beta_triplet(d, rho_o, cfg)
        sig.append(blending_function(beta, ba, bs, cfg.chi))
    check("d > ρ_b+ρ_o+ρ_a ⇒ σ = 1", abs(sig[0] - 1.0) < 1e-12, f"σ = {sig[0]:.4f}")
    check("ρ_s 内 ⇒ σ = 0", abs(sig[2]) < 1e-12, f"σ = {sig[2]:.4f}")
    check("混合区 ⇒ 0 < σ < 1", 0.0 < sig[1] < 1.0, f"σ = {sig[1]:.4f}")


def t6_gpio() -> None:
    print("\nT6  Theorem 2：GPIO 对多项式型扰动的渐近收敛（式 (23)(24)）")
    dt = 0.001
    for m, desc in ((2, "常值扰动"), (3, "常值+斜坡扰动")):
        gpio = GPIO(GPIOConfig(order=m, omega_o=8.0), x0=0.0)
        x = 0.0
        for k in range(int(6.0 / dt)):
            t = k * dt
            v = 0.6 * np.sin(0.5 * t)                       # 已知输入（如 r）
            d_true = 0.4 + (0.1 * t if m >= 3 else 0.0)     # 被估计的扰动
            x_dot = v + d_true
            x += dt * x_dot
            gpio.step(x, v, dt)
        err = abs(gpio.disturbance - d_true)
        check(f"m={m}（{desc}）：d̂ → d，6 s 后误差 < 5e-3", err < 5e-3,
              f"d̂={gpio.disturbance:.5f}, d={d_true:.5f}, err={err:.2e}")


def t7_smc() -> None:
    print("\nT7  Theorem 3：滑模面 s 有限时间收敛（式 (27)(29)-(31)）")
    dt = 0.001
    lam, k3, k4 = 1.0, 1.0, 0.5
    d_p = 0.35                     # 常值合成扰动（小于 k4，满足 k4 > Δ）

    # (a) 理想 sign 切换：严格有限时间收敛
    s = 5.0
    converged_at = None
    for k in range(int(8.0 / dt)):
        t = k * dt
        u = (-k3 * s - k4 * np.sign(s)) / 1.0        # den = cosψ+λsinψ = 1
        s += dt * (1.0 * u + d_p)
        if converged_at is None and abs(s) < 1e-3:
            converged_at = t
            break
    check("sign 切换：s 有限时间收敛到 0 邻域", converged_at is not None,
          f"|s|<1e-3 于 t = {converged_at:.3f} s（理论上界 |s0|/(k4−Δ) = {5.0/(k4-d_p):.1f} s）")
    check("k4 > |Δ| 的有限时间前提成立", k4 > abs(d_p), f"k4 = {k4}, Δ = {d_p}")

    # (b) tanh 去抖（论文建议）：收敛到边界层内的准滑模残差
    eps = 0.01
    s = 5.0
    for _ in range(int(10.0 / dt)):
        u = (-k3 * s - k4 * np.tanh(s / eps)) / 1.0
        s += dt * (1.0 * u + d_p)
    # 理论准滑模残差：−k3 s − k4 tanh(s/eps) + d_p = 0 ⇒ s ≈ d_p/(k3 + k4/eps)
    predicted = d_p / (k3 + k4 / eps)
    check("tanh 去抖：残差落在边界层内且与解析预测一致",
          abs(s) < 10 * eps and abs(s - predicted) < 0.25 * predicted + 1e-3,
          f"s = {s:.5f}，解析预测 = {predicted:.5f}")


def t8_kinematics() -> None:
    print("\nT8  运动学模型（式 (1)）解析一致性")
    r = Unicycle(0.0, 0.0, 0.0)
    for _ in range(1000):
        r.step(1.0, 0.0, (0.0, 0.0, 0.0), 0.01)
    check("u=1, r=0：直线运动 x = 10 m", abs(r.x - 10.0) < 1e-9, f"x = {r.x:.9f}")
    r2 = Unicycle(0.0, 0.0, 0.0)
    omega = 0.5
    T = 4.0
    for _ in range(int(T / 0.01)):
        r2.step(1.0, omega, (0.0, 0.0, 0.0), 0.01)
    R = 1.0 / omega
    check("u=1, r=0.5：圆周运动半径 = u/r = 2 m",
          abs(np.hypot(r2.x - R * np.sin(omega * T), r2.y - R * (1 - np.cos(omega * T)))) < 1e-6,
          f"位置 = ({r2.x:.6f}, {r2.y:.6f})，解析 = ({R*np.sin(omega*T):.6f}, {R*(1-np.cos(omega*T)):.6f})")


def t9_singularity() -> None:
    print("\nT9  式 (27) 分母奇异性与保护（缺口#6）")
    cfg = GuidanceConfig(lam=1.0, den_guard=0.12)
    from mgvfdc.controllers import MGVFDCController
    ctrl = MGVFDCController(GVFConfig(), cfg)
    # 取 ψ 使 cosψ + λ sinψ = 0 → ψ = −π/4
    psi = -np.pi / 4.0
    ctrl.reset(np.array([0.0, 0.0]), psi)
    out = ctrl.step(eta_b=np.array([20.0, 20.0]), psi=psi, psi_dot=0.0,
                    eta_t=np.array([0.0, 0.0]), obstacle_pos=[], obstacle_rho=[],
                    dt=0.002)
    check("奇异点处指令仍有限且满足 u ∈ [u_min, u_max]",
          np.isfinite(out.u) and cfg.u_min <= out.u <= cfg.u_max,
          f"u = {out.u:.4f} (u_raw = {out.u_raw:.2e})")


def t10_blend_product() -> None:
    """勘误回归测试：式 (12) 的吸引项权重是**连乘 ∏**，不是求和 Σ。

    语义：障碍物只"削弱/切断"目标的吸引力，绝不产生吸引力。
    连乘的关键性质：只要有任意一个障碍物使 σ_i = 0，整个吸引项立刻归零。
    本实现早期因 PDF 提取器丢掉了连乘号字形（productdisplay），误读为 Σ 并在多障碍
    场景放大了吸引项 → 机器人穿入障碍。此测试把正确读法永久钉住。
    """
    print("\nT10  式 (12) 吸引项是连乘 ∏（勘误回归测试）")

    cfg = GVFConfig(blend="product", rep_sense_rule="side")
    eta_b = np.array([3.0, 0.0])
    eta_t = np.array([0.0, 0.0])
    # 障碍 1：近到只剩斥力（与机器人相距 2.9 m < ρ_b+ρ_o+ρ_s = 3.1 m ⇒ σ=0）
    # 障碍 2、3：远到纯吸引（σ=1）
    near = np.array([3.0 + 2.9, 0.0])
    far1 = np.array([100.0, 0.0])
    far2 = np.array([0.0, 100.0])
    pos = [near, far1, far2]
    rho = [1.0, 1.0, 1.0]

    F_prod, diag = resultant_field(eta_b, eta_t, pos, rho, cfg)
    sig = diag.sigmas
    check("近障碍 σ = 0、远障碍 σ = 1", abs(sig[0]) < 1e-12 and abs(sig[1] - 1) < 1e-12,
          f"σ = {np.round(sig, 4)}")

    # 手工构造"只有斥力、没有任何吸引"的对照场
    expected = np.zeros(2)
    for i, (p, r) in enumerate(zip(pos, rho)):
        sense = circulation_sense(eta_b, eta_t, p, cfg.rep_sense_rule)
        fr = repulsive_field(eta_b, p, sense) * cfg.rep_strength
        n = float(np.linalg.norm(fr))
        if n > 1e-12:
            fr = fr / n
        expected = expected + (1.0 - sig[i]) * fr

    err = float(np.linalg.norm(F_prod - expected))
    check("∏σ_i = 0 ⇒ 吸引项被完全切断（只剩斥力兜底）", err < 1e-12,
          f"‖F_∏ − 纯斥力对照‖ = {err:.2e}")

    # 误读的求和读法会把吸引项额外注入（Σσ_i = 2）
    F_sum, _ = resultant_field(eta_b, eta_t, pos, rho, GVFConfig(blend="sum"))
    amp = float(np.linalg.norm(F_sum - expected))
    check("Σσ_i 读法会额外注入吸引力（错误来源已定位）", amp > 1e-6,
          f"Σ 读法相对纯斥力场的偏离 {amp:.2e}（∏ 读法为 {err:.2e}）")

    # 另一种等价稳健写法：min_i σ_i，同样满足"任一为 0 则整体为 0"
    F_min, _ = resultant_field(eta_b, eta_t, pos, rho, GVFConfig(blend="min"))
    d = float(np.linalg.norm(F_min - F_prod))
    check("min_i σ_i 与 ∏σ_i 在本例中一致（等价的稳健写法）", d < 1e-12,
          f"‖F_min − F_∏‖ = {d:.2e}")


# ----------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("论文理论性质的数值验证  ——  MGVFDC 复现")
    print("=" * 78)
    for fn in (t1_theorem1, t2_detour, t3_tangential, t4_blending,
               t5_sigma_semantics, t6_gpio, t7_smc, t8_kinematics, t9_singularity,
               t10_blend_product):
        fn()
    print("\n" + "=" * 78)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
