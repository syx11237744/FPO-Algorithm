import os
from omnisafe.utils.path import RESULT_PATH
from omnisafe.utils.visualizer import extract_training_data_with_type, get_statistics_with_type, ENVTITLES


envs = [
    'SafetyPointGoal1-v0',
    'SafetyPointPush1-v0',
    'SafetyPointButton1-v0',
    'SafetyCarPush1-v0',
    'SafetyCarButton1-v0',
    'SafetyCarCircle1-v0',
]

tags = [
    'error_0',
    'error_20',
    'error_40',
]

TAGNAMES = {
    'error_0': 'Step 0',
    'error_20': 'Step 20',
    'error_40': 'Step 40',
}

extract_training_data_with_type(
    envs=envs,
    algs=['FPO'],
    tags=tags,
    alg_type='error',
)
df = get_statistics_with_type(
    envs=envs,
    algs=['FPO-error'],
    tags=tags,
    last=1.0,
)

pivot_df = df.pivot(
    index='env',
    columns=['tag'],
    values=['mean', 'ci'],
)

# print(pivot_df)
# exit()

markdown_lines = []

header = ['Environment']
for tag in tags:
    header.extend([TAGNAMES[tag], ''])
markdown_lines.append('|' + '|'.join(header) + '|')

separator = ['-'] * (1 + len(tags))
markdown_lines.append('|' + '|'.join(separator) + '|')

sums = {}
counts = {}
for env in envs:
    row = [ENVTITLES[env]]
    for tag in tags:
        key = tag
        mean_val = pivot_df['mean'].loc[env, key]
        row.append(f'{mean_val:.2f}')
        sums[key] = sums.get(key, 0) + mean_val
        counts[key] = counts.get(key, 0) + 1
    markdown_lines.append('|' + '|'.join(row) + '|')

avg_row = ['**Average**']
for tag in tags:
    key = tag
    avg_val = sums[key] / counts[key]
    avg_row.append(f'{avg_val:.2f}')
markdown_lines.append('|' + '|'.join(avg_row) + '|')

with open(os.path.join(RESULT_PATH, 'error.md'), 'w') as f:
    f.write('\n'.join(markdown_lines))
