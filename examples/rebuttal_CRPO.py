import os

from omnisafe.utils.path import RESULT_PATH
from omnisafe.utils.visualizer import extract_training_data, get_statistics, normalize_by, ENVTITLES


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
    'SafetyHumanoidVelocity-v1',
    'SafetyHalfCheetahVelocity-v1',
    'SafetyHopperVelocity-v1',
    'SafetyWalker2dVelocity-v1',
    'SafetySwimmerVelocity-v1',
]

algs = [
    'OnCRPO',
    'FPO',
]
baseline = 'PPO'

ALGNAMES = {
    'OnCRPO': 'CRPO',
    'FPO': 'FPO',
    'PPO': 'PPO',
}

tags = [
    'cost',
    'return',
]

TAGNAMES = {
    'cost': 'Cost',
    'return': 'Return',
}

# extract_training_data(envs, algs + [baseline], tags)

df = get_statistics(envs, tags, algs + [baseline], to_file=False)

normalize_by(df, baseline)

pivot_df = df.pivot(
    index='env',
    columns=['alg', 'tag'],
    values=['mean', 'ci'],
)

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
            mean_val = pivot_df['mean'].loc[env, key]
            row.append(f'{mean_val:.3f}')
            sums[key] = sums.get(key, 0) + mean_val
            counts[key] = counts.get(key, 0) + 1
    markdown_lines.append('|' + '|'.join(row) + '|')

avg_row = ['**Average**']
for alg in algs:
    for tag in tags:
        key = (alg, tag)
        avg_val = sums[key] / counts[key]
        avg_row.append(f'{avg_val:.3f}')
markdown_lines.append('|' + '|'.join(avg_row) + '|')

with open(os.path.join(RESULT_PATH, 'CRPO.md'), 'w') as f:
    f.write('\n'.join(markdown_lines))
