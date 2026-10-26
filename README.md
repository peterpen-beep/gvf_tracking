# MGVFDC 复现：四足机器人无碰撞目标跟踪控制器（含不确定扰动抑制）

**从零实现**论文：

> S. Kong, J. Sun, A. Luo, W. Chi, C. Zhang, S. Zhang, Y. Liu, Q. Zhu, J. Yu,
> "A Collision-Free Target Tracking Controller With Uncertain Disturbance Rejection for Quadruped Robots,"
> *IEEE Transactions on Intelligent Vehicles*, vol. 9, no. 1, pp. 670–680, Jan. 2024.
> DOI: [10.1109/TIV.2023.3296669](https://doi.org/10.1109/TIV.2023.3296669)

本目录**不依赖任何已有工程代码**（与工作区 `code/` 无关），只用 `numpy` + `matplotlib` 独立实现论文第 III~V 节的
全部方法，并复现其三个仿真算例与对比结论。

---

## 1. 快速开始

```bash
# 环境：Python ≥ 3.9，仅需 numpy / matplotlib
pip install -r requirements.txt

# 理论性质数值验证（24 项，约 5 秒）
python tests/test_verification.py

# 完整复现（生成全部图 + 指标表，约 2~5 分钟）
python run_all.py

# 缩减时长快速跑通（约 20 秒，数值不用于正式结论）
python run_all.py --quick
```

产物：

| 路径 | 内容 |
|------|------|
| `outputs/figures/*.png` | 论文 Fig.2~Fig.9 的复刻图 + 验证/消融图 |
| `outputs/metrics.csv` | 所有算例的全部指标 |
| `outputs/summary.txt` | 人类可读的结果汇总 |
| `outputs/run_log.txt` | 最近一次运行日志 |

---

## 2. 目录结构

```
gvf_tracking/
├── README.md                       # 本文档
├── CODE_GUIDE.md                   # ★ 代码文件指南（以 Case 1 为主线，含行号索引与"改哪里"速查表）
├── PAPER_GUIDE.md                  # ★ 论文讲解（零基础版：预备知识 → 公式推导 → 难点 → 实现坑）
├── run_all.py                      # 一键复现入口（编排全部实验与绘图）
├── requirements.txt
├── src/mgvfdc/
│   ├── config.py                   # 全部参数（论文给定值 + 整定值与依据）
│   ├── gvf.py                      # 论文式 (4)-(20)：改进制导向量场
│   ├── observer.py                 # 论文式 (23)(28)：GPIO 广义比例积分观测器
│   ├── controllers.py              # 论文式 (21)(22)(26)(27)(32)(33)：制导控制器
│   ├── baselines.py                # 对比基线 GVFPID（引文 [31] 的方法学重建）
│   ├── plant.py                    # 论文式 (1)：独轮车运动学 + 目标 + 圆形障碍
│   ├── scenarios.py                # Case 1/2/3 与 Fig.3/Fig.5 场景
│   ├── sim.py                      # 闭环仿真引擎
│   ├── metrics.py                  # RMSE / 安全距离 / 路径长度 / 到达时间
│   └── plotting.py                 # 全部绘图
├── tests/
│   ├── test_verification.py        # 9 组 24 项理论性质数值验证
│   ├── smoke_sim.py                # 快速排查脚本（三算例 × 四控制器）
│   ├── debug_trace.py              # 逐时刻内部量追踪（调试用）
│   └── design_case3.py             # Case 3 窄门宽度标定 + 扰动灵敏度
├── tools/
│   └── make_video.py               # 演示视频生成（MP4，独立于 src/）
├── .vendor/                        # 工程内安装的 imageio-ffmpeg（含 ffmpeg 7.1 二进制）
└── outputs/
    ├── figures/                    # 静态图
    ├── videos/                     # 演示视频 + 关键帧拼图
    ├── metrics.csv / summary.txt   # 指标与结论
    └── run_log.txt / video_log.txt # 运行日志
```

---

## 3. 论文公式 → 代码映射表

| 论文公式 | 内容 | 代码位置 |
|---|---|---|
| 式 (1) | 独轮车运动学模型 η̇ = R(ψ)[u,v,r]ᵀ + d_w，v ≡ 0 | `plant.py: Unicycle.step`（RK4） |
| 式 (2)(3) | 跟踪目标 / 避障目标 | `metrics.py`（RMSE、安全距离指标） |
| 式 (4) | 基础向量场 F = ε(pᵀη)η − p(ηᵀη) | `gvf.py: attractive_field / repulsive_field` |
| 式 (5) | 制导角 ψ_r = arctan2(F_y, F_x) | `gvf.py: resultant_field` |
| 式 (6)(7) | 吸引场（ε=2） | `gvf.py: attractive_field` |
| 式 (8) | **改进 1**：可变属性向量 p = −η_e | `gvf.py: attractive_field(att_mode="variable")` |
| 式 (9) | Theorem 1 指向性 | `tests/test_verification.py: t1_theorem1`（数值核验） |
| 式 (10)(11) | 斥力场（ε=1） | `gvf.py: repulsive_field` |
| 式 (12) | 合成场 | `gvf.py: resultant_field` |
| 式 (13) | 融合权重 σ_i(β_i) | `gvf.py: blend_weight` |
| 式 (14)(15)(16) | β_i, β_a,i, β_s,i | `gvf.py: beta_triplet` |
| 式 (17)-(20) | **改进 2**：可调融合函数 ξ(β) | `gvf.py: blending_function` |
| 式 (21) | 转向误差 ė_ψ = ψ̇ − ψ̇_r | `controllers.py: MGVFDCController.step` |
| 式 (22) | 转向律 r = −k₁e_ψ − k₂ė_ψ − d̂_w3 | 同上 |
| 式 (23) | 偏航通道 GPIO | `observer.py: GPIO` |
| 式 (24)(25) | GPIO 误差动态与 Hurwitz 条件 | `observer.py`（极点配置 ω_j = C(m,j)ω_o^j） |
| 式 (26) | 趋近用降阶模型 | `plant.py` |
| 式 (27) | 滑模趋近律 u = (−k₃s − k₄sign(s) − d̂_p)/(cosψ+λsinψ) | `controllers.py`（tanh 替代 sign） |
| 式 (28) | 滑模通道 GPIO | `observer.py` |
| 式 (29)-(31) | Theorem 3 有限时间收敛 | `tests/test_verification.py: t7_smc` |
| 式 (32)(33) | 转向优先策略 u ← exp(ψ_T−\|e_ψ\|)·u | `controllers.py`（含 `priority_gain` 记录） |

---

## 4. 复现缺口：论文未给出、必须自行确定的部分

以下 7 项是论文的"空白"，本实现全部给出**有依据的取值**并在代码注释中标注 `【整定】`。
另附 1 项**我自己的勘误**（#2，非论文问题）：

| # | 缺口 | 本实现的选择 | 依据 |
|---|------|--------------|------|
| 1 | 斥力场属性向量 p **未给出** | p_i = sense·[−y_e,oi, x_e,oi] | 由式 (10)(11) 可证 ε=1 时 F_r·η ≡ 0（切向场），取 p ⊥ η 得到恒定旋向的纯切向场；绕行方向由"保持同侧"叉积规则确定 |
| 2 | **【勘误·我读错了，不是论文的问题】** 式 (12) 吸引项是**连乘 ∏**，我曾误读为求和 Σ | 默认 `blend="product"`：吸引项权重 = ∏_i σ_i（论文原式） | 论文原文前半部分是 **∏**（障碍物只"削弱/切断"目标吸引力，绝不产生吸引）。我最初用自写 PDF 提取器时，连乘号字形（`productdisplay`，CMEX10 字体）不在映射表里被**静默丢弃**，才误读为 Σ。后果：多障碍时吸引项被放大而穿入障碍（**−0.383 m**）；改回 ∏ 后 **+2.050 m**。勘误对照见 `outputs/figures/fig9_blend_gap.png`，回归测试见 `tests/test_verification.py` 的 T10，审计脚本见 `.pdftools/audit_glyphs.py` |
| 3 | ψ_T（转向优先阈值）未给数值 | 0.45 rad | 论文自称"由工程经验确定"；0.45 rad ≈ 26°，兼顾"先转向"与"不过度减速" |
| 4 | 全部控制器增益 k₁~k₄、λ、ω₁ⱼ、ω₂ⱼ、GPIO 阶数 m 未给 | k₁=2, k₂=0.6, k₃=1, k₄=0.5, λ=1, m=3, ω_o=8/6 rad/s | 用极点配置公式 ω_j = C(m,j)ω_o^j 把观测器极点配置在 −ω_o；k₂<1 由离散稳定性决定（见 E3） |
| 5 | 式 (22) 存在**代数环**（ė_ψ 含 r） | 用测量角速度 ψ̇_meas 代替 | 实现问题 E1 |
| 6 | 式 (27) 分母 cosψ+λsinψ **会过零** | 奇异保护 + u/r 饱和 | 实现问题 E2（不处理会使 u 发散到 1e2 量级） |
| 7 | 动态目标速度前馈缺失（式 (29) 忽略了 ẋ_t） | 不显式前馈，把它并入集总扰动由 GPIO 估计 | GPIO 内部模型为 ≤(m−1) 次多项式，可无差跟踪常速目标带来的常值偏置 |
| 8 | 论文无开源代码、障碍物布局与目标相位未给 | 障碍物按"必须真正触发避障"原则布置（接近走廊 + 目标轨迹附近），目标相位照抄论文公式 | 见 `scenarios.py` 注释 |

### 4.1 实现在论文之外必须补的 4 个工程问题

| 编号 | 问题 | 现象与解决 |
|---|---|---|
| **E1** | 式 (22) 的 ė_ψ 同时出现在等式两端（代数环） | 用 ψ̇_meas 替代，闭环化为 ė_ψ = [−k₁e_ψ + (d_w3−d̂_w3) − ψ̇_r]/(1+k₂)，一阶稳定 |
| **E2** | 式 (27) 分母 cosψ+λsinψ 在 ψ ≈ arctan(−1/λ) 时过零 | 分母钳位（`den_guard=0.12`）+ u ∈ [u_min, u_max] 饱和；否则 u 会瞬时放大到 1e2 m/s |
| **E3** | **ψ̇ 的获取方式决定稳定性** | 若用航向差分 ψ̇ ≈ (ψ_k−ψ_{k−1})/dt，−k₂ψ̇ 项会引入一步延迟，离散闭环变成 r_k = −k₂r_{k−1}+…，当 \|k₂\| ≥ 1 时发散（实测 r 在 ±1.5 rad/s 之间高频抖振、机器人绕圈丢失目标）。真实 IMU 直接给出角速度，无此延迟；本实现采用 IMU 式角速度测量，并取 k₂ = 0.6 < 1 |
| **E4** | 式 (27) 的 u **结构上有符号** | 在"机头对准目标"的正常工况下分子分母异号，u 自动为正；但固定 p 的经典 GVF 会把参考航向指向背离目标的一侧，此时 u<0 才是正确的。四足 Remark 1 只禁止横荡输入 v ≡ 0，故默认取 u ≥ 0（先转向再前进，与转向优先策略哲学一致）；把 `u_min` 设为 −u_max 可得到有符号变体，用于复现固定 p 的闭环绕行轨迹。**强制 u ≥ 0 时固定 p 会出现 u = 0 死锁**（实测仅前移 2.57 m，见 `summary.txt`） |

---

## 5. 复现结果

### 5.1 理论性质的数值验证（`python tests/test_verification.py`）

```
通过 24 项，失败 0 项
```

| 验证项 | 结果 |
|---|---|
| Theorem 1：可变属性向量下 F_a 与 −η_e 夹角 | max\|1−cos Θ_e\| = 4.4e-16（2000 个随机位形） |
| 推论 ‖F_a‖ = ‖η_e‖³ | 125.000000 vs 125.000000 |
| 改进 1 的几何效果：从 S 点积分曲线长度 | 固定 p 7.060 m vs 可变 p 2.080 m（直线 2.121 m） |
| 斥力场切向性 F_r·η ≡ 0 | max\|cos\| = 1.9e-16；沿场积分 20 m 半径漂移 < 1e-9 m |
| 融合函数 ξ：端点值/端点导数/接缝值 1−χ/C¹ 连续/单调 | 5 个 χ 值全部通过 |
| Theorem 2：GPIO 对常值与斜坡扰动收敛 | 6 s 后误差 2.7e-4 / 9.3e-5 |
| Theorem 3：滑模面有限时间收敛 | sign 切换 3.527 s 收敛（理论上界 33.3 s） |
| tanh 去抖的准滑模残差 | 0.00835 vs 解析预测 0.00686 |
| 式 (1) 解析一致性（直线/圆周） | 误差 < 1e-9 |

### 5.2 论文 Fig.3：改进 1（可变属性向量）消除绕远路

| 指标 | 固定 p（经典 GVF） | 可变 p = −η_e（改进 1） |
|---|---|---|
| S 点积分曲线长度（论文几何：目标[5,5]、起点[6.5,6.5]） | 7.03 m | **2.07 m** |
| 闭环路径长度（远程场景，直线距离 28.284 m） | 34.06 m | **28.284 m（零绕行）** |
| 到达目标 1 m 邻域时刻 | 27.55 s | **22.74 s** |

见 `fig1_field_comparison.png`、`fig1b_detour_closedloop.png`。

### 5.3 论文 Fig.4 / Fig.5：改进 2（可调融合函数 χ）

χ 增大 ⇒ 机器人与障碍边界的距离缩短（论文结论，已复现）：

| χ | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|
| 最小安全距离 [m] | 2.122 | 2.070 | 1.992 | 1.928 | **1.887** |

### 5.4 论文 Fig.6 / Fig.8：Case 1（动态目标 + 静态多障碍）

| 方法 | 稳态 RMSE x / y [m] | 路径长度 [m] | 最小安全距离 [m] |
|---|---|---|---|
| **MGVFDC（完整方法）** | **0.0001 / 0.0002** | 70.2 | **2.050**（> ρ_b ✓） |
| GVFPID（基线 [31]） | 4.0742 / 7.0739 | 135.3 | 1.886 |
| 参考：关闭避障（纯追踪） | 0.0001 / 0.0002 | 66.9 | **−0.647（穿入障碍）** |

- 复现了论文"GVFPID 轨迹 much more tortuous、耗时更长"的结论（135.3 m vs 70.2 m）；
- 关闭避障的参考实验证明**避障机制确实是必要的**（安全距离 −0.647 m，即撞穿障碍）；
- 论文报告该算例稳态 RMSE 为 0.075 / 0.091 m（实机水平），本实现的理想运动学模型为 1e-4 m 级；
  加入 2 cm 位置噪声 + 0.5° 航向噪声 + 底层速度残差后为 **0.0482 / 0.0425 m**，与论文同量级。

### 5.5 论文 Fig.7：Case 2（动态目标 + 动态障碍）

| 方法 | 路径长度 [m] | 最小安全距离 [m] |
|---|---|---|
| MGVFDC | 71.8 | 1.478（> ρ_b ✓） |
| GVFPID | 131.3 | 1.925 |

### 5.6 论文 Fig.9：Case 3（含不确定扰动 d_w = [−0.1, 0.05, 0.4]ᵀ）

| 方法 | 最小安全距离 [m] | 制导角 RMSE [rad] | 稳态 RMSE x / y [m] | 末端距目标 [m] |
|---|---|---|---|---|
| **MGVFDC（有 GPIO 补偿）** | **1.201**（> ρ_b ✓） | 0.392 | **0.0002 / 0.0002** | **0.0003** |
| MGVF（无补偿） | **0.654（违反判据）** | 0.343 | 31.12 / 14.00 | 36.22（被障碍困住） |
| GVFPID（基线） | 1.089 | 0.578 | 0.155 / 0.114 | 0.193 |

复现了论文"有扰动补偿才能有效避免碰撞"的核心结论：无补偿时稳态航向偏置 e_ss = d_w3/k₁ = 0.2 rad，
导致机器人贴障碍通过（安全距离跌破 ρ_b）并被障碍"困住"、始终无法抵达目标。

### 5.7 附加实验：扰动灵敏度（论文只给了单点 d_w3 = 0.4）

| d_w3 [rad/s] | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| MGVFDC 最小安全距离 [m] | 1.202 | 1.202 | 1.201 | 1.201 | 1.200 | 1.196 |
| MGVFDC 稳态 RMSE [m] | 0.0003 | 0.0003 | 0.0003 | 0.0004 | 0.0004 | 0.0004 |
| MGVF 最小安全距离 [m] | 1.202 | 0.932 | 0.654 | 0.361 | **−0.049** | **−0.324** |
| MGVF 稳态 RMSE [m] | 0.0132 | 22.56 | 23.21 | 23.62 | 23.89 | 24.05 |
| GVFPID 最小安全距离 [m] | 1.267 | 1.278 | 1.089 | 0.771 | 0.169 | 0.839 |

结论：GPIO 补偿使系统在 **d_w3 直到 1.0 rad/s（论文工况的 2.5 倍）** 仍保持 0.0004 m 级跟踪与
> ρ_b 的安全裕度；无补偿版本在 d_w3 ≥ 0.8 时安全距离转负（真实碰撞）。

---

## 6. 与论文的差异说明（诚实清单）

1. **底层动力学**：论文把 [u, 0, r] 交给引文 [6] 的凸 MPC + 摆动腿 + 地面反力控制器；本复现按其运动学模型
   式 (1) **直接积分理想速度**，把底层误差建模为 d_w。这是论文自身的建模层次（其仿真 Case 1/2 的 d_w = 0），
   因此本复现的跟踪精度优于论文实机数值（见 5.4 末段）。
2. **符号输入**：默认 u ≥ 0；需要 `u_min=−u_max` 才能复现固定 p 的闭环绕行（E4）。
3. **斥力场标度**：论文未给属性向量标度（缺口#1），本实现在 `normalize_fields=True` 下把两个场单位化后融合。
   实测（Case 1）：归一化时最小安全距离 **2.050 m**，不归一化时 **1.344 m**——**安全性在两种情况下都成立**
   （因为连乘 ∏ 会在 σ=0 处把吸引项彻底切断），归一化改善的是轨迹质量与安全余量。
4. **式 (12) 的吸引项权重**：严格执行论文原式**连乘 ∏σ_i**（默认 `blend="product"`）。
   本实现早期曾把它误读为求和 Σ（原因是我自写的 PDF 提取器丢掉了连乘号字形），
   导致多障碍场景吸引项被放大而穿入障碍（−0.383 m）；改回 ∏ 后为 +2.050 m。
   误读版本仍保留为 `blend="sum"` 仅供消融对照，对照图见 `fig9_blend_gap.png`。
5. **GVFPID 基线**：引文 [31] 的实现细节未在本文复述，本实现按其方法学要点重建（经典固定 p 场 + PID 制导），
   并对 PID 做了抗积分饱和处理，以保证对比公平。
6. **未实现部分**：论文的实机实验（IMU + 腿部里程计 / RTK-GPS、坑洼草坪踢击测试）不在本复现范围内；
   对应的仿真替代是 5.4 的"加噪现实水平"实验。
7. **动态障碍速度上限**：论文自述"动态障碍不能太快"，本仓库提供 `stress_dynamic_obstacle`（障碍 0.5 m/s
   横向扫过）用于验证该局限。

---

## 7. 扩展接口（便于后续改进）

```python
import mgvfdc as M

# 自定义 GVF 参数与控制器增益
gvf = M.GVFConfig(rho_a=3.0, rho_s=1.5, chi=0.5, blend="product")   # 论文原式：连乘 ∏σ_i
ctrl = M.GuidanceConfig(k1=3.0, k2=0.5, lam=1.2, psi_T=0.6,
                        yaw_obs=M.GPIOConfig(order=3, omega_o=12.0),
                        s_obs=M.GPIOConfig(order=2, omega_o=8.0))
scn = M.case3(duration=90.0, gate_offset=2.4)
res = M.run_simulation(scn, M.MGVFDCController(gvf, ctrl), M.SimConfig(dt=0.001))
print(M.summarize(res))

# 只画某个场，不做仿真
F, diag = M.resultant_field(eta_b=[10, 3], eta_t=[0, 0],
                            obstacle_pos=[[5, 0]], obstacle_rho=[1.0], cfg=gvf)
print(diag.psi_r, diag.sigmas)
```

可调旋钮速查：`GVFConfig`（ρ_b/ρ_a/ρ_s/χ/绕行规则/单位化/加权读法）、
`GuidanceConfig`（k₁~k₄/λ/ψ_T/u、r 饱和/GPIO 阶数与带宽/补偿开关）、
`SimConfig`（步长/时长/扰动/测量噪声）、`scenarios.py`（场景几何）。

---

## 8. 视频可视化（MP4）

`tools/make_video.py` 是**独立的可视化脚本**，不改动 `src/mgvfdc/` 任何实现：它调用仿真引擎跑出轨迹，
再逐帧渲染成视频。

```bash
python tools/make_video.py                 # 生成全部 3 段视频
python tools/make_video.py --only case3    # 只生成某一段（case1 / case3 / fig5）
python tools/make_video.py --preview       # 只渲染 30 帧快速自检
python tools/make_video.py --fps 30 --dt-frame 0.08 --crf 20   # 调帧率/采样/画质
```

| 产物 | 内容 |
|---|---|
| `outputs/videos/case1_panorama.mp4` | Case 1 全景：6 个障碍 + 动态目标，MGVFDC 的完整接近与避障过程 |
| `outputs/videos/case3_comparison.mp4` | Case 3 三方法**并排**对比：MGVFDC 穿过窄门 / MGVF 被卡住 / GVFPID 绕远 |
| `outputs/videos/fig5_chi_sweep.mp4` | Fig.5 场景：5 个 χ 并排，安全距离曲线 + 运行中的历史最小安全距离柱状图 |
| `outputs/videos/*_poster.png` | 每段视频的 2×2 关键帧拼图（不播视频也能快速看结论） |

画面内容：左侧为战场视图（GVF 单位向量场背景、障碍物 + 红色虚线安全圆 ρ_b+ρ_o、目标星标与轨迹、
机器人三角标记 + 实际航向箭头 + 参考航向 ψ_r 箭头、轨迹拖尾、实时状态文字）；
右侧为同步时序曲线（距目标距离、最小安全距离（含 ρ_b 判据线）、控制输入 u）。

**MP4 编码依赖**：本机原本没有 ffmpeg，已把 `imageio-ffmpeg`（自带 ffmpeg 7.1 二进制）安装到工程内的
`.vendor/`，脚本启动时会自动把它加入 `sys.path`，**不污染系统的 conda 环境**；若删掉 `.vendor/`，
脚本会自动退回输出 GIF。

**关于"机器人没有腿"**：论文的模型是式 (1) 的二维独轮车抽象（机器人 = 平面上的点 + 朝向角），
底层动力学外包给引文 [6]，因此视频中的机器人是带航向的三角标记。若要看到真正的四足机器人行走视频，
需要接入 MuJoCo/MJX 的 XG 模型与已训练的 PPO 策略（即"制导层 + 学习型底层"的集成），属于另一量级的工作。
