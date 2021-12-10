The code is designed to work with LunarLanderContinuous-v2 from OpenAI Gym.
Please install OpenAI Gym environment on your computer first and then run this software.

LQT control algorithm is implemented in 'heuristic' function in the lunar_hopper_LQT.py.

RL is a reinforcement learning code, and LQT-TRO is a LQT control with learning-based reference trajectory optimization.
To get both codes working, lunar_lander.py in the box2d of gym files must be replaced with lunar_hopper_LQT.py.

For more details about each algorithm, please read the following article:
T. Tanaka, H. Malki and M. Cescon. Linear Quadratic Tracking with Reinforcement Learning Based Reference Trajectory Optimization for the Lunar Hopper in Simulated Environment. IEEE Access

Note1: RL and LQT-TRO codes are based on an open-source code developed by Clayton Thorrez (https://github.com/cthorrez/ml-fun/tree/master/ddpg). Special thanks to him.

Note2: the code is still experimental, and might have some bugs. Any inquiry and feedback should go to ttanaka@uh.edu.

Thanks!

tskTNK
