import gym
import torch
import time

import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy.sparse import linalg

global y_target_latest
y_target_latest = 0

def demo_ddpg_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    mu = torch.load('log/lander/models/model_7980')
    mu = mu.eval()

    while True:

        s = torch.tensor(s)
        a = torch.squeeze(mu(s)).data.numpy()

        FPS = 50
        SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
        VIEWPORT_W = 600
        VIEWPORT_H = 400

        gravity = 9.8/FPS/FPS # gravity is enhanced by scaling
        thrust_main_max = gravity/0.56
        thrust_side_max = thrust_main_max*0.095/0.7 # m/frame^2 # determined by test
        m_main_inv = thrust_main_max    # gravity*0.57
        m_side_inv = thrust_side_max    # gravity*0.225
        a_i_inv= 0.198/100 # rad/frame^2 # determined by test # not depend on SCALE
        align = 0.87   # 0.87 = sin30

        # target point set
        x_target = 0
        y_target = 0   # the landing point is 0
        Vx_target = 0
        Vy_target = 0
        theta_target = 0
        omega_target = 0

        """
        change the x_target&y_target for three different phases
        - Approaching phase: x and y are changed
        - Landing phase: only y is changed (vertical descent)
        - Free fall phase: free fall
        """

        for i in range(2):
            if a[i] < env.action_space3.low[i]:
                a[i] = env.action_space3.low[i]
            elif a[i] > env.action_space3.high[i]:
                a[i] = env.action_space3.high[i]

        k1 = float(a[0])*2 + 2 # range: 2 - 6
        k2 = float(a[1]) # range: 0 - 2
        Hl_target = 0.1

        # Approaching phase
        """
        # LQT was successful with the following parameters
        k1 = 2
        k2 = 2
        Hl_target = 0.1
        """

        x_target = s[0]*(VIEWPORT_W/SCALE/2)/k1
        y_target1 = Hl_target*(VIEWPORT_H/SCALE/2) # obstacle avoidance ydirection
        y_target2 = k2*(VIEWPORT_H/SCALE/2)*(0.4-abs(s[8])) # obstacle avoidance xdirection
        y_target_new = y_target1 + y_target2 + (s[1]-s[9])*(VIEWPORT_H/SCALE/2)
        # comparing with the last value and update only when the new value is larger
        global y_target_latest
        if y_target_new > y_target_latest:
            y_target_latest = y_target_new
        y_target = y_target_latest

        # Landing phse <- fixed
        if abs(s[0]) < 0.05:
            y_target = s[1]*(VIEWPORT_H/SCALE/2)/1.6

        X = np.array([ \
        [s[0]*(VIEWPORT_W/SCALE/2)-x_target], \
        [s[1]*(VIEWPORT_H/SCALE/2)-y_target], \
        [s[2]/(VIEWPORT_W/SCALE/2)-Vx_target], \
        [s[3]/(VIEWPORT_H/SCALE/2)-Vy_target], \
        [s[4]-theta_target], \
        [s[5]/20.0-omega_target]])

        # print("X {}\n".format(X))

        A = np.array([ \
        [0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, -1*gravity, 0], \
        [0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0]])

        B = np.array([ \
        [0, 0], \
        [0, 0], \
        [0, m_side_inv*align], \
        [1*m_main_inv, 0], \
        [0, 0], \
        [0, -1*a_i_inv]])

        sigma = np.array([ \
        [0], \
        [0], \
        [0], \
        [-1*gravity], \
        [0], \
        [0]])

        # gravity compensation
        BTB = np.dot(B.T, B)
        u_sigma = -1*np.linalg.inv(BTB).dot(B.T).dot(sigma)
        # print("u_sigma {}\n".format(u_sigma))

        # Design of LQR
        # Solve Riccati equation to find a optimal control input
        R = np.array([ \
        [1, 0], \
        [0, 1]])

        Q = np.array([ \
        [1, 0, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0], \
        [0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 100, 0], \
        [0, 0, 0, 0, 0, 100]])

        # Solving Riccati equation
        P = sp.linalg.solve_continuous_are(A, B, Q, R)
        # print("P {}\n".format(P))

        # u = -KX
        # K = R-1*Rt*P
        K = np.linalg.inv(R).dot(B.T).dot(P)
        thrust = -1*np.dot(K, X) + u_sigma

        BK = np.dot(B, K)
        A_ = A - BK
        a_eig = np.linalg.eig(A_)
        a_sort = np.sort(a_eig[0])
        # print("eigen values {}\n".format(a_sort))

        # print("thrust {}\n".format(thrust))
        # thrust[0] = 0
        # thrust[1] = 1

        # free fall phase
        if abs(s[0]) < 0.05 and s[1] < 0.01:
            thrust[0] = 0
            thrust[1] = 0

        # conversion to compensate main thruster's tricky thrusting
        thrust[0] = thrust[0]/0.5-1.0

        a_updated = np.array([thrust[0], thrust[1]])
        a_updated = np.clip(a_updated, -1, +1)  #  if the value is less than 0.5, it's ignored
        # print("a_updated * {}\n".format(a_updated))

        # s, r, done, info = env.step(a)
        s, r, done, info = env.step(a_updated)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 1 == 0 or done:
        # if done:
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("step {} reward {:+0.2f}".format(steps, total_reward),"observations:", " ".join(["{:+0.2f}".format(x) for x in s]), "a_updated:","{} {}".format(a_updated[0], a_updated[1]), "actions:","{} {}".format(a[0], a[1]))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    time.sleep(3)

    for i in range(1):
        demo_ddpg_lander(env, seed=i, render=True)
