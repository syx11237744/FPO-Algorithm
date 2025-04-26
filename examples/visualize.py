import os

import omnisafe.utils.path as path
from omnisafe.utils.visualizer import extract_training_data, get_statistics, draw_cost_return_scatter,plot_threshold_bargraph

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
    'RCPO',
    'PPOLag',
    'FOCOPS',
    'P3O',
    'FPO',
]

tags = [
    'cost',
    'return',
]

extract_training_data(envs, algs, tags)
get_statistics(envs, tags)
plot_threshold_bargraph(os.path.join(path.RESULT_PATH, 'convergence_statistics.csv'))
draw_cost_return_scatter(os.path.join(path.RESULT_PATH, 'statistics.csv'), normalize_by='PPOLag')
