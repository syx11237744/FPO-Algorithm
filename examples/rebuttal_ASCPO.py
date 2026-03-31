import os
from omnisafe.utils.path import RESULT_PATH
from omnisafe.utils.visualizer import extract_training_data_with_type, get_statistics_with_type, ENVTITLES


envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointPush1-v0',
    'SafetyPointButton1-v0',
    'SafetyPointCircle1-v0',
    'SafetyCarGoal1-v0',
    'SafetyCarPush1-v0',
    'SafetyCarButton1-v0',
    'SafetyCarCircle1-v0',
    'SafetyAntVelocity-v1',
    'SafetyHalfCheetahVelocity-v1',
    'SafetyHopperVelocity-v1',
    'SafetyHumanoidVelocity-v1',
    'SafetySwimmerVelocity-v1',
    'SafetyWalker2dVelocity-v1',
]

extract_algs = [
    'PPO',
    'FPO',
]

algs = [
    'ASCPO',
    'FPO',
]

normalize_by = 'PPO'

ALGNAMES = {
    'ASCPO': 'ASCPO',
    'FPO': 'FPO',
}

ASCPO_DATA = {
    'SafetyPointGoal1-v0': {'cost': 3.35, 'return': 1.61},
    'SafetyPointPush1-v0': {'cost': 5.89, 'return': -1.05},
    'SafetyPointButton1-v0': {'cost': 9.35, 'return': -1.20},
    'SafetyPointCircle1-v0': {'cost': 26.99, 'return': 1.30},
    'SafetyCarGoal1-v0': {'cost': 3.10, 'return': -0.03},
    'SafetyCarPush1-v0': {'cost': 6.25, 'return': -0.03},
    'SafetyCarButton1-v0': {'cost': 18.63, 'return': -1.34},
    'SafetyCarCircle1-v0': {'cost': 7.53, 'return': 0.53},
    'SafetyAntVelocity-v1': {'cost': 0.24, 'return': -263.68},
    'SafetyHalfCheetahVelocity-v1': {'cost': 0.02, 'return': 186.62},
    'SafetyHopperVelocity-v1': {'cost': 0.20, 'return': 286.15},
    'SafetyHumanoidVelocity-v1': {'cost': 0.01, 'return': 173.73},
    'SafetySwimmerVelocity-v1': {'cost': 3.30, 'return': -18.54},
    'SafetyWalker2dVelocity-v1': {'cost': 0.01, 'return': 46.50},
}

tags = [
    'cost',
    'return',
]

TAGNAMES = {
    'cost': 'Cost',
    'return': 'Return',
}

import pandas as pd


df = extract_training_data(
    envs=envs,
    algs=algs,
    tags=tags
)

df = get_statistics_with_type(
    envs=envs,
    algs=['PPO', 'FPO'],
    tags=tags
)

ascpo_rows = []
for env in envs:
    for tag in tags:
        ascpo_rows.append({
            'env': env,
            'alg': 'ASCPO',
            'tag': tag,
            'mean': ASCPO_DATA[env][tag],
            'ci': 0.0
        })
ascpo_df = pd.DataFrame(ascpo_rows)
df = pd.concat([df, ascpo_df], ignore_index=True)

pivot_df = df.pivot(
    index='env',
    columns=['alg', 'tag'],
    values=['mean', 'ci'],
)

normalized_df = pivot_df.copy()
for env in envs:
    for tag in tags:
        ppo_value = pivot_df['mean'].loc[env, (normalize_by, tag)]
        for alg in ['ASCPO', 'FPO']:
            if ppo_value != 0:
                normalized_df['mean'].loc[env, (alg, tag)] = pivot_df['mean'].loc[env, (alg, tag)] / ppo_value
            else:
                normalized_df['mean'].loc[env, (alg, tag)] = pivot_df['mean'].loc[env, (alg, tag)]

markdown_lines = []

header = ['Environment']
for alg in algs:
    header.extend([ALGNAMES[alg], ''])
markdown_lines.append('|' + '|'.join(header) + '|')

separator = ['-'] * (1 + len(algs) * len(tags))
markdown_lines.append('|' + '|'.join(separator) + '|')

subheader = ['']
for alg in algs:
    for tag in tags:
        subheader.append(TAGNAMES[tag])
markdown_lines.append('|' + '|'.join(subheader) + '|')

sums = {}
counts = {}
for env in envs:
    row = [ENVTITLES[env]]
    for alg in algs:
        for tag in tags:
            key = (alg, tag)
            mean_val = normalized_df['mean'].loc[env, key]
            row.append(f'{mean_val:.2f}')
            sums[key] = sums.get(key, 0) + mean_val
            counts[key] = counts.get(key, 0) + 1
    markdown_lines.append('|' + '|'.join(row) + '|')

avg_row = ['**Average**']
for alg in algs:
    for tag in tags:
        key = (alg, tag)
        avg_val = sums[key] / counts[key]
        avg_row.append(f'{avg_val:.2f}')
markdown_lines.append('|' + '|'.join(avg_row) + '|')

with open(os.path.join(RESULT_PATH, 'ASCPO.md'), 'w') as f:
    f.write('\n'.join(markdown_lines))
