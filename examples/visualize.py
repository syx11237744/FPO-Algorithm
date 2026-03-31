import os
import numpy as np
import matplotlib.pyplot as plt
import omnisafe.utils.path as path
from omnisafe.utils.visualizer import \
    extract_training_data, get_statistics, get_table, \
    plot_cost_return_scatter, plot_training_curve, plot_legend


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointPush1-v0',
    'SafetyPointCircle1-v0',
    'SafetyCarGoal1-v0',
    'SafetyCarPush1-v0',
    'SafetyCarCircle1-v0',
    'SafetyAntVelocity-v1',
    'SafetyHumanoidVelocity-v1',
    'SafetyPointButton1-v0',
    'SafetyCarButton1-v0',
    'SafetyHalfCheetahVelocity-v1',
    'SafetyHopperVelocity-v1',
    'SafetyWalker2dVelocity-v1',
    'SafetySwimmerVelocity-v1',
]

algs = [
    'PPO',
    'CPO',
    'PCPO',
    'FOCOPS',
    'RCPO',
    'PPOLag',
    'TRPOPID',
    'P3O',
    'FPO',
]

tags = [
    'cost',
    'return',
]

# extract_training_data(envs, algs, tags)
# get_statistics(envs, tags, algs)
# get_table()
# step = np.linspace(int(2e4), int(1e7), 200)
# plot_training_curve(envs, algs, tags, step)
# plot_legend()
# plot_cost_return_scatter(envs, algs)