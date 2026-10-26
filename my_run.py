import numpy as np
from math import comb


def gvf(robot_pos, target_pos, obstacles):
    # 机器狗相对目标
    ne = robot_pos - target_pos

    # 吸引场F_a
    ne_dot = np.dot(ne, ne)
    F_a = ne_dot * (-ne)

    # 【补全坑1】归一化吸引场（教程缺口#1，防止吸引力淹没排斥力）
    norm_Fa = np.linalg.norm(F_a)
    if norm_Fa > 1e-8:
        F_a = F_a / norm_Fa

    # 初始化总排斥场和吸引力权重
    F_r = np.zeros(2)
    pre_sigma = 1.0

    # 遍历障碍物
    for obs in obstacles:
        obs_pos = obs[:2]
        obs_r = obs[2]

        ne_obs = robot_pos - obs_pos
        dist_obs = np.dot(ne_obs, ne_obs)

        # 式子（14）
        beta_i = obs_r ** 2 - dist_obs

        # 式（15）（16）
        robot_safe = 1.0  # 机器狗安全半径
        #这里激活和安全直接所有障碍物都一样了，但是后期可以每个都定义
        dist_open = 2.1  # 激活
        dist_end = 1.1  # 安全
        beta_a = obs_r ** 2 - (robot_safe + dist_open + obs_r) ** 2
        beta_s = obs_r ** 2 - (robot_safe + dist_end + obs_r) ** 2

        # 式（19）（20）
        chis = 0.5  # 调节参数
        delta_beta = beta_s - beta_a
        divide = beta_a + chis * delta_beta

        if beta_i <= beta_a:
            sigma = 1
        elif beta_i <= divide:
            # 19
            sigma = 1.0 - (beta_i - beta_a) ** 2 / (chis * delta_beta ** 2)
        elif beta_i <= beta_s:
            # 20
            sigma = (beta_i - beta_s) ** 2 / ((1 - chis) * delta_beta ** 2)
        else:
            sigma = 0.0

        # 排斥场
        F_r_i = dist_obs * np.array([ne_obs[1], -ne_obs[0]])

        # # 【补全坑1】归一化排斥场
        norm_Fr = np.linalg.norm(F_r_i)
        if norm_Fr > 1e-8:
            F_r_i = F_r_i / norm_Fr

        F_r += (1 - sigma) * F_r_i
        pre_sigma *= sigma

    F = pre_sigma * F_a + F_r

    fai = np.arctan2(F[1], F[0])

    return fai



def gpio(input_u, measure_state, dt, obs_state, omega, m):
    fai_hat = obs_state[0]
    d_hat = obs_state[1:]

    # 预测误差
    e = fai_hat - measure_state

    # 求导
    d_hat_dot = np.zeros(m)
    for i in range(m - 1):
        d_hat_dot[i] = d_hat[i + 1] - omega[i+1] * e
    d_hat_dot[-1] = -omega[-1] * e

    state_hat_dot = input_u + d_hat[0] - omega[0] * e

    # 欧拉积分
    fai_hat += dt * state_hat_dot
    d_hat += dt * d_hat_dot

    new_obs_state = np.concatenate(([fai_hat], d_hat))
    return d_hat[0], new_obs_state


def init_gpio(measure_state, m, omega_o):
    # 二项增益omega
    omega = np.array([comb(m, j) * (omega_o ** j) for j in range(1, m + 1)])
    # 初始状态
    obs_state = np.zeros(m + 1)
    obs_state[0] = measure_state
    return obs_state, omega


def main():
    # 参数
    params = {
        'm': 3,#观测器阶数
        'omega_o': 8.0,#观测器带宽
        'k1': 2.0, 'k2': 0.6,#转向pd增益
        'k3': 1.0, 'k4': 0.5,#滑膜增益
        'lambda_val': 1.0,#滑膜面权重
        'epsilon': 1.0, #tanh过渡带
        'fai_T': 0.5,#转向优先阈值
        'u_max': 1.5,#最大转向速度
    }

    #仿真设置
    dt=0.002
    steps=20000

    #初始化机器狗 坐标朝向
    robot_state=np.array([0.0,0.0,0.0])

    #目标和障碍
    target_pos=np.array([10.0,10.0])
    obstacles = [
        np.array([3.0, 1.6, 0.6]),
        np.array([5.6, 6.9, 0.8]),
        np.array([8.2, 2.6, 0.6])
    ]  # [x, y, r] 六个静态障碍

    #扰动误差dw
    dw=np.array([-0.1,0.05,0.4])

    #初始化观测器  转向观测器，趋近观测器 omega相同
    steering_obs_state,omega=init_gpio(robot_state[2],params['m'],params['omega_o'])
    approach_obs_state,_=init_gpio(0.0,params['m'],params['omega_o'])

    #上一时刻角度，速度  和参考航向
    r_pre,u_pre=0.0,0.0

    fai_r = gvf(robot_state[:2], target_pos, obstacles)
    fai_r_pre = fai_r

    for t in range(steps):
        x,y,fai=robot_state

        #向量场gvf
        fai_r=gvf(robot_state[:2],target_pos,obstacles)

        #解决pi跳变：计算归一化角度差
        diff = fai_r - fai_r_pre
        diff = (diff + np.pi) % (2 * np.pi) - np.pi  # 把差值强行映射到 [-pi, pi]
        fai_r_dot = diff / dt

        fai_r_pre=fai_r

        #模拟IMU测量 航向角速度：角速度就等于“下发的指令 r”加上“环境打滑 dw3”
        fai_dot_meas = r_pre + dw[2]

        #转向gpio更新   转向
        dw3_hat,steering_obs_state=gpio(r_pre,fai,dt,steering_obs_state,omega,params['m'])

        #转向控制  式（22）
        e_fai=fai-fai_r
        r=-params['k1']*e_fai-params['k2']*(fai_dot_meas-fai_r_dot)-dw3_hat
        r = np.clip(r, -2.0, 2.0)  # 限制方向盘打死

        #计算滑模面s
        x_e=x-target_pos[0]
        y_e=y-target_pos[1]
        s=x_e+params['lambda_val']*y_e

        #gpio更新   趋近
        input_u=(np.cos(fai)+params['lambda_val']*np.sin(fai))*u_pre
        d_p_hat,approach_obs_state=gpio(input_u,s,dt,approach_obs_state,omega,params['m'])

        #趋近控制律 u_raw (式27)
        den=np.cos(fai)+params['lambda_val']*np.sin(fai)

        if abs(den)<0.12:
            den=0.12 if den>0 else -0.12
        u_raw = (-params['k3'] * s - params['k4'] * np.tanh(s / params['epsilon']) - d_p_hat) / den

        #转向优先策略
        if abs(e_fai)>params['fai_T']:
            u=np.exp(params['fai_T']-abs(e_fai))*u_raw
        else:
            u=u_raw


        #速度限制 （保证四足只前进不后退，且不超速）
        u=np.clip(u,0.0,params['u_max'])

        #积分运动学模型 式（1）
        x_dot=u*np.cos(fai)+dw[0]
        y_dot=u*np.sin(fai)+dw[1]
        fai_dot=r+dw[2]

        #跟随机器狗的移动
        robot_state[0]+=x_dot*dt
        robot_state[1]+=y_dot*dt
        robot_state[2]+=fai_dot*dt


        #更新
        r_pre,u_pre=r,u

        # 目标移动 圆周运动 (速度0.4m/s)
        target_pos = [10.0 + 2 * np.sin(0.2 * t * dt), 10.0 + 2 * np.cos(0.2 * t * dt)]

        # 日志
        if t % 500 == 0:
            #到目标距离
            dist_to_target = np.sqrt((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2)

            print(f"Step {t:6d} | 位置: ({x:6.2f}, {y:6.2f}) | 距离目标: {dist_to_target:5.2f} m | 航向: {fai:6.2f} | "
                  f"参考航向: {fai_r:6.2f} | dw3估计: {dw3_hat:6.3f} | "
                  f"dp估计: {d_p_hat:6.3f} | 速度u: {u:4.2f} | 角速度r: {r:5.2f}")

if __name__ == "__main__":
    main()
