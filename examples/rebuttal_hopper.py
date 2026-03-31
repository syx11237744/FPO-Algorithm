from omnisafe.utils.visualizer import extract_training_data


envs = [
    'SafetyHopper-v4',
]

algs = [
    'CPO',
    'CUP',
    'FOCOPS',
    'PPOLag',
    'RCPO',
]

tags = [
    'cost',
    'return',
]

extract_training_data(envs, algs, tags)
