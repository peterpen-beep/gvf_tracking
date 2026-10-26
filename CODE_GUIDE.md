# 代码文件指南（以 Case 1 为主线）

> 本指南回答两个问题：**① 跑 Case 1 到底用了哪些文件；② 每个文件是干什么的、想改什么该去哪里。**
> 全文的 `文件:行号` 都对应当前代码版本，可直接跳转。
> （本指南为新增文档，未改动任何 `.py` 实现代码。）

---

## 1. 先看这个：跑 Case 1 其实只要 4 个文件

```python
import sys; sys.path.insert(0, r"D:\workpalce\gvf_tracking\src")
import mgvfdc as M                                    # ① __init__.py 包门面

scn  = M.case1(120.0)                                 # ② scenarios.py  场景（起点/目标/障碍）
ctrl = M.MGVFDCController(M.GVFConfig(),              # ③ controllers.py 控制器
                          M.GuidanceConfig())         # ④ config.py     参数
res  = M.run_simulation(scn, ctrl, M.SimConfig())     # ⑤ sim.py        闭环仿真引擎
print(M.summarize(res))                               # ⑥ metrics.py    指标
```

> 上面这段已实测：60 s 仿真耗时约 10 秒，输出 `末端距目标 0.0001 m | 稳态RMSE 0.0002/0.0002 | 最小安全距离 2.050 m`。
> `mgvfdc` 包里其余文件（`gvf.py`、`observer.py`、`plant.py`）是被 ③⑤ 间接调用的，你不需要直接 import。

---

## 2. 分层结构：谁依赖谁

```
第 7 层  呈现     run_all.py · tools/make_video.py · plotting.py
                                   │  （把轨迹变成图/视频）
第 6 层  评估     metrics.py       │  （RMSE/安全距离/路径长度…）
                                   ▼
第 5 层  组装     scenarios.py（场景长什么样） + sim.py（一步怎么走）
                                   │
第 4 层  控制     controllers.py（论文方法） · baselines.py（对比基线）
                                   │
第 3 层  被控对象 plant.py（式(1) 运动学 + 目标 + 障碍物）
                                   │
第 2 层  数学     gvf.py（式(4)-(20) 向量场） · observer.py（式(23)(28) GPIO）
                                   │
第 1 层  参数     config.py（ρ_b/ρ_a/ρ_s/χ/增益/步长…所有旋钮）
                                   │
第 0 层  依赖     numpy · matplotlib · （视频另需 .vendor/ 里的 imageio-ffmpeg）
```

**记忆口诀**：`config` 给参数 → `gvf`+`observer` 算数学 → `plant` 做物理 → `controllers` 下指令 →
`scenarios`+`sim` 把它们装成一次完整实验 → `metrics` 算成绩 → `plotting`/`make_video` 出图出视频 → `run_all` 一键跑完。

---

## 3. Case 1 的完整调用链（带行号）

```
run_all.py:156   [4/10] Case 1：动态目标 + 静态多障碍
│
├─ run_all.py:157   sc1 = M.case1(dur)                          ← 生成场景
│    └─ scenarios.py:83   case1()
│         ├─ scenarios.py:70  _corridor_obstacles()  布置 6 个障碍（沿"接近走廊"）
│         │    ├─ scenarios.py:41  _polar()          极角→直角坐标
│         │    └─ plant.py:123     Obstacle(...)     圆形障碍物对象
│         └─ scenarios.py:47  _dynamic_target()      论文的圆周动态目标
│              └─ plant.py:78      CircularTarget(...)  x=30+15sin(0.021t−π/6) 等
│
├─ run_all.py:159   M.run_simulation(sc1, MGVFDCController(gvf, ctrl), sim)
│    │
│    ├─ controllers.py:58   MGVFDCController.__init__()
│    │    └─ observer.py:42   GPIO(...) × 2            偏航通道 + 滑模通道观测器
│    │         └─ config.py:32  gains                  极点配置 ω_j = C(m,j)·ω_o^j
│    │
│    ├─ controllers.py:75   reset()                    观测器初值、绕行方向缓存清零
│    │
│    └─ sim.py:71   run_simulation()  主循环（dt=0.002 s × 60001 步）
│         │
│         ├─ plant.py:94/99   target.position(t)       目标当前时刻位置
│         ├─ plant.py:137     Obstacle.position(t)     障碍物当前时刻位置
│         │
│         ├─ controllers.py:106  MGVFDCController.step(...)   ★ 核心，一步内的全部控制
│         │    ├─ gvf.py:152   resultant_field()       ★ 合成 GVF 场 → 参考航向 ψ_r
│         │    │    ├─ gvf.py:33   attractive_field()  式(6)(7)(8)：p = −η_e 的可变属性向量
│         │    │    ├─ gvf.py:74   beta_triplet()      式(14)(15)(16)：β_i, β_a,i, β_s,i
│         │    │    ├─ gvf.py:101  blend_weight()      式(13)：融合权重 σ_i
│         │    │    │    └─ gvf.py:82  blending_function()  式(19)(20)：可调融合函数 ξ(β)
│         │    │    ├─ gvf.py:109  circulation_sense() 绕障碍物的旋向（保持同侧）
│         │    │    └─ gvf.py:54   repulsive_field()   式(10)(11)：切向斥力场
│         │    ├─ controllers.py:84  _resolve_senses() 进入激活区后锁定旋向（防抖动）
│         │    ├─ observer.py:56   yaw_gpio.step()     式(23) 偏航扰动估计 d̂_w3
│         │    ├─ observer.py:56   s_gpio.step()       式(28) 滑模通道扰动估计 d̂_p
│         │    └─ 四段控制律：转向(式22) → 趋近(式27) → 转向优先(式32/33) → 饱和
│         │
│         ├─ plant.py:45    Unicycle.step()            式(1) RK4 积分一步
│         ├─ plant.py:167   safety_clearance()         算到各障碍物边界的距离
│         └─ sim.py:22      SimResult                  把每一步的所有量存进日志数组
│
├─ run_all.py:160   M.make_controller("gvfpid", ...)            对比基线
│    └─ controllers.py:187 → baselines.py:39  GVFPIDController（经典固定 p 场 + PID）
│
├─ run_all.py:162-163  关闭避障参考（rep_strength=0）          证明"避障机制是必要的"
│
├─ run_all.py:164   plotting.fig_case1(...)      → plotting.py:268   论文 Fig.6（轨迹 + 误差）
├─ run_all.py:170   M.summarize(res)             → metrics.py:57     论文全部指标
├─ run_all.py:183   plotting.fig_safety(...)     → plotting.py:343   论文 Fig.8（安全距离曲线）
└─ run_all.py:232   plotting.fig_blend_gap(...)  → plotting.py:456   勘误对照：∏ 连乘（论文原式）vs Σ 求和（早期误读）

视频版（可选）：
tools/make_video.py:356  build_case1()  →  :149 render()  →  :94 field_background()（复用 gvf.resultant_field 画场）
                                                            :115 robot_triangle()（画机器人）
                                                            :133 arena_limits()（算视野范围）
                                                            :340 montage()（拼关键帧预览图）
```

---

## 4. 逐文件说明

### 4.1 `src/mgvfdc/` —— 核心库（Case 1 真正干活的都在这里）

| 文件 | 一句话职责 | 关键内容（行号） | Case 1 用吗 |
|---|---|---|---|
| `config.py` | **所有参数旋钮**（论文给定值 + 整定值，注释里标了来源） | `GVFConfig`:42、`GuidanceConfig`:70、`GPIOConfig`:19、`SimConfig`:104 | ✅ |
| `gvf.py` | **向量场数学**（论文第 III 节，式 4–20） | `attractive_field`:33、`repulsive_field`:54、`beta_triplet`:74、`blending_function`:82、`circulation_sense`:109、`resultant_field`:152、`FieldDiagnostics`:134 | ✅ |
| `observer.py` | **GPIO 观测器**（论文式 23、28） | `GPIO`:31（`step`:56 推进一拍、`disturbance`:91 取 d̂） | ✅（被控制器调用） |
| `plant.py` | **被控对象**：式(1) 运动学 + 目标 + 障碍物 | `Unicycle.step`:45（RK4 积分式 1）、`CircularTarget`:78、`Obstacle`:123、`safety_clearance`:167 | ✅ |
| `controllers.py` | **论文的制导控制器**（论文第 IV 节，式 21–33） | `MGVFDCController.step`:106、`ControlOut`:34、`make_controller`:187 | ✅ |
| `baselines.py` | **对比基线 GVFPID**（论文引文 [31] 方法学重建） | `GVFPIDController`:39、`PIDGains`:29 | ✅（对比用） |
| `scenarios.py` | **场景库**：每个算例的起点/目标/障碍几何 | `case1`:83、`case2`:99、`case3`:128、`fig5_scenario`:200 | ✅（Case 1 入口） |
| `sim.py` | **闭环仿真引擎**：把控制器接到模型上跑完 | `run_simulation`:71、`SimResult`:22、`run_comparison`:176 | ✅ |
| `metrics.py` | **评价指标** | `summarize`:57、`rmse`:19、`path_length`:35、`arrival_time`:42、`steady_mask`:28 | ✅ |
| `plotting.py` | **静态图**（论文各图的复刻） | `fig_case1`:268、`fig_safety`:343、`fig_blend_gap`:456、`save`:49 | ✅（3 个图） |
| `__init__.py` | 包门面，把上面所有东西导出成 `M.xxx` | — | ✅ |

### 4.2 顶层脚本

| 文件 | 一句话职责 | 关键内容 | Case 1 用吗 |
|---|---|---|---|
| `run_all.py` | **一键复现编排**：依次跑 10 个实验段落、出全部图、写 `metrics.csv` / `summary.txt` | `run_all()`:91（Case 1 在 :155-190）、`main()`:309 | ✅（Case 1 在第 4~5 段） |
| `tools/make_video.py` | **视频生成**（独立于核心库，只调用不修改） | `build_case1`:356、`render`:149、`field_background`:94、`make_writer`:78 | ✅（Case 1 视频） |

### 4.3 测试与调试工具（不参与出结果，但排查问题时最有用）

| 文件 | 用途 | 什么时候用 |
|---|---|---|
| `tests/test_verification.py` | 24 项理论性质验证（Theorem 1/2/3、融合函数 C¹、斥力场切向性…） | 改完 `gvf.py`/`observer.py` 后**必跑**，确认数学没写坏 |
| `tests/smoke_sim.py` | 三个算例 × 四个控制器的快速对照打印 | 改完参数想快速看效果 |
| `tests/debug_trace.py` | 逐时刻打印内部量（`x y psi psi_r e_psi u r s den sig_min`） | 机器人行为异常时定位是哪一环坏了 |
| `tests/design_case3.py` | Case 3 窄门宽度标定 + 扰动灵敏度扫描 | 设计新场景时参考 |

### 4.4 数据与依赖目录

| 路径 | 内容 |
|---|---|
| `outputs/figures/*.png` | 16 张静态图（`run_all.py` 产出） |
| `outputs/videos/*.mp4` | 3 段演示视频 + 关键帧拼图（`tools/make_video.py` 产出） |
| `outputs/metrics.csv` | 全部算例指标表（可用 Excel 打开） |
| `outputs/summary.txt` | 人类可读的结论汇总（**先看这个**） |
| `outputs/run_log.txt` / `video_log.txt` | 最近一次运行的完整日志 |
| `.vendor/` | 工程内安装的 `imageio-ffmpeg`（自带 ffmpeg 二进制），删掉则视频脚本自动退回 GIF |

---

## 5. Case 1 一步（0.002 s）内部到底发生了什么

对应 `controllers.py:106` 的 `step()`，**严格按代码顺序**：

| 序 | 代码位置 | 做什么 | 对应论文 |
|---|---|---|---|
| 1 | `controllers.py:116-117` | 算位置误差 `η_e = η_b − η_t` | — |
| 2 | `controllers.py:121-127` → `gvf.py:152` | **算合成场 F 与参考航向 ψ_r**（两遍：先探 σ 以决定是否锁定旋向） | 式 (4)–(12) |
| 3 | `gvf.py:33` | 吸引场（p = −η_e，可变属性向量） | 式 (6)(7)(8) |
| 4 | `gvf.py:74` + `gvf.py:82` | 逐障碍物算 β 与融合权重 σ | 式 (13)–(16)(19)(20) |
| 5 | `gvf.py:109` + `controllers.py:84` | 定绕行旋向（仅在未激活时更新，进入后锁定） | 缺口#1 |
| 6 | `gvf.py:54` + `gvf.py:195-205` | 切向斥力场，按 (1−σ) 加权求和 | 式 (10)(11)(12) |
| 7 | `gvf.py:217-227` | 吸引项权重：**连乘 ∏σ_i（论文原式，默认）**；另有 `"min"` 等价写法与 `"sum"`（早期误读，仅对照） | 式 (12) |
| 8 | `controllers.py:130-135` | ψ_r 数值微分 + 限幅，得 ψ̇_r | — |
| 9 | `controllers.py:137-143` | **转向律** r = −k₁e_ψ − k₂(ψ̇−ψ̇_r) − d̂_w3，再限幅 | 式 (21)(22) |
| 10 | `controllers.py:145-146` | 用**实际下发**的 r 推进偏航 GPIO | 式 (23) |
| 11 | `controllers.py:148-153` | **滑模面** s = x_e + λy_e，分母奇异保护 | 式 (26)(27) |
| 12 | `controllers.py:155-157` | **趋近律** u = (−k₃s − k₄tanh(s/ε) − d̂_p)/(cosψ+λsinψ) | 式 (27) |
| 13 | `controllers.py:159-165` | **转向优先**：\|e_ψ\|>ψ_T 时 u ← exp(ψ_T−\|e_ψ\|)·u，再饱和 | 式 (32)(33) |
| 14 | `controllers.py:167-168` | 用**实际下发**的 u 推进滑模通道 GPIO | 式 (28) |
| 15 | `controllers.py:170-180` | 打包成 `ControlOut` 返回 | — |
| 16 | `sim.py:151` → `plant.py:45` | 用 (u, r) 与扰动 d_w 积分模型一步 | 式 (1) |
| 17 | `sim.py:129` → `plant.py:167` | 算安全距离，连同其余量写入 `SimResult` 各数组 | 式 (3) |

---

## 6. `SimResult` 数据字典（`sim.py:22`）

跑完一次仿真拿到的对象，27 个字段 + 3 个方法。画图、做指标全靠它：

| 字段 | 形状 | 含义 |
|---|---|---|
| `t` | (T,) | 时间轴 [s] |
| `x`, `y`, `psi` | (T,) | 机器人真实位姿（世界系位置 + 偏航角） |
| `u`, `r` | (T,) | 实际下发的纵向速度/偏航角速度指令 |
| `psi_r` | (T,) | GVF 制导角（参考航向） |
| `s` | (T,) | 滑模面 s = x_e + λy_e |
| `e_psi` | (T,) | 转向误差 ψ − ψ_r |
| `d_w3_hat`, `d_p_hat` | (T,) | GPIO 的扰动量估计（**看观测器工作是否正常就靠它**） |
| `u_raw`, `priority_gain` | (T,) | 未饱和指令 / 转向优先缩放系数（诊断饱和与减速） |
| `sigma_min` | (T,) | 当前最小融合权重（=0 表示曾进入纯斥力区） |
| `tgt_x`, `tgt_y` | (T,) | 目标轨迹 |
| `err_x`, `err_y`, `dist` | (T,) | 位置误差与到目标距离 |
| `clearance` | (n_obs, T) | **各障碍物的安全距离指标** d_i − ρ_o,i（论文 Fig.8 的纵轴） |
| `obstacle_xy` | (n_obs, 2, T) | 障碍物轨迹（动态障碍画图用） |
| `obstacle_rho` | (n_obs,) | 各障碍物半径 |
| 方法 | — | `n_obs` 属性、`min_clearance()`、`safety_margin()`（全局最小安全距离） |

---

## 7. 「我想改 X → 去改哪个文件」速查表

| 你的需求 | 改哪里 | 具体旋钮 |
|---|---|---|
| 换障碍物位置/半径/数量 | `scenarios.py:70` `_corridor_obstacles()` | `(沿路径距离, 横向偏移, 半径)` 三元组列表 |
| 换机器人起点/初始朝向 | `scenarios.py:83` `case1()` | `start=(x,y)`、`psi0=...` |
| 换目标轨迹 | `scenarios.py:47` `_dynamic_target()` | 圆心/半径/角速度/相位 |
| 改避障激活距离 ρ_a、最小安全距离 ρ_s | `config.py:46-47` | `rho_a`、`rho_s` |
| 改融合函数可调参数 χ | `config.py:48` | `chi` |
| 改控制器增益（快/慢、抖不抖） | `config.py:73-79` | `k1 k2 k3 k4 lam eps_tanh` |
| 改转向优先阈值 ψ_T | `config.py:80` | `psi_T` |
| 改速度上限/下限（是否能倒车） | `config.py:88-89` | `u_max`、`u_min` |
| 改 GPIO 阶数与带宽（扰动估计快慢） | `config.py:96-97`、`config.py:28-29` | `GPIOConfig(order=m, omega_o=…)` |
| 改仿真步长/时长/扰动/噪声 | `config.py:107-111` | `dt`、`duration`、`d_w`、`meas_noise_std` |
| 改融合函数公式本身 | `gvf.py:82` `blending_function()` | 式 (19)(20) 的实现 |
| 换吸引场/斥力场形式 | `gvf.py:33` / `gvf.py:54` | 式 (6)(7)(8) / 式 (10)(11) |
| 改式 (12) 的吸引项权重读法 | `config.py:68` + `gvf.py:217-227` | `blend = "product"`（论文原式，连乘 ∏σ_i，默认）/ `"min"`（等价写法）/ `"sum"`（早期误读，仅对照） |
| 加新指标（如平均功率、抖动度） | `metrics.py:57` `summarize()` | 往返回字典里加键 |
| 加新图 | `plotting.py` 里仿照 `fig_case1`:268 写一个，再在 `run_all.py` 里调用 | — |
| 加新算例 | `scenarios.py` 加函数 + `run_all.py` 加一段 | 参考 `case3`:128 |
| 改视频画面/布局/时长 | `tools/make_video.py:149` `render()` | `--fps`、`--dt-frame`、`--crf` |

---

## 8. Case 1 用到的 / 没用到的

**Case 1 用到**（✅）：
`config.py` · `gvf.py` · `observer.py` · `plant.py`（`Unicycle`/`CircularTarget`/`Obstacle`/`safety_clearance`）·
`controllers.py` · `baselines.py`（GVFPID 对比）· `scenarios.py:83 case1` · `sim.py` · `metrics.py` ·
`plotting.py` 的 `fig_case1`/`fig_safety`/`fig_blend_gap` · `run_all.py:155-190, 232` · `tools/make_video.py`（视频）

**Case 1 不涉及**（➖，别在这几个文件里找 Case 1 的逻辑）：
`scenarios.py` 的 `case2`/`case3`/`fig3_*`/`fig5_scenario`/`stress_dynamic_obstacle`；
`plant.py` 的 `StaticTarget`/`LinearTarget`（那是 Fig.3/Fig.5 场景用的）；
`plotting.py` 的 `fig_framework`/`fig_field_comparison`/`fig_detour_closedloop`/`fig_blending`/
`fig_chi_tuning`/`fig_case2`/`fig_case3`/`fig_verification`/`fig_disturbance_sweep`；
`tools/make_video.py` 的 `build_case3`/`build_fig5`/`render_chi_sweep`/`draw_obstacles`（未被调用）。

---

## 9. 建议的阅读顺序（跟着数据流走，最省力）

1. **`config.py`** —— 先知道有哪些旋钮、默认值多少（10 分钟，全部是数据类，好读）；
2. **`scenarios.py` 的 `case1()`** —— 看 Case 1 长什么样：起点、目标怎么动、6 个障碍在哪（配合 `_corridor_obstacles()`）；
3. **`gvf.py`** —— 按 `attractive_field → repulsive_field → beta_triplet → blending_function → resultant_field` 的顺序读，每个函数头部都写了对应论文公式；
4. **`observer.py`** —— 只有 100 行，`GPIO.step()` 就是论文式 (23) 的逐行翻译；
5. **`controllers.py` 的 `step()`** —— **全文最该精读的函数**（约 80 行），对照第 5 节的 16 步表逐行看；
6. **`plant.py` 的 `Unicycle.step()`** —— 看被控对象只有 3 行状态方程；
7. **`sim.py` 的 `run_simulation()`** —— 看主循环怎么把上面这些串起来（注意看它记录了哪些量）；
8. **`metrics.py` + `plotting.py:fig_case1`** —— 看结果怎么变成论文里的图和表；
9. **`tests/debug_trace.py`** —— 最后拿去当"显微镜"用：改参数后跑一次，逐时刻看内部量。

跑一遍：

```powershell
cd D:\workpalce\gvf_tracking
D:\anaconda\python.exe tests\test_verification.py    # 先确认数学没坏（24 项）
D:\anaconda\python.exe tests\debug_trace.py          # 看 Case 1 逐时刻内部量
D:\anaconda\python.exe tests\smoke_sim.py            # 三算例 × 四控制器对照
```
