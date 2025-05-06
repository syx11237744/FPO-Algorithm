import os
from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
from omnisafe.utils.path import LOG_PATH, RESULT_PATH, FIGURE_PATH


EXTENSION = 'png'

TAGLOGNAMES = {
    'cost': 'Metrics/EpCost',
    'return': 'Metrics/EpRet',
}

TAGLABELS = {
    'cost': 'Episode cost',
    'return': 'Episode return',
}

ALGLABELCOLORS = {
    'CPO': ('CPO', 'C0'),
    'PCPO': ('PCPO', 'C1'),
    'FOCOPS': ('FOCOPS', 'C2'),
    'RCPO': ('RCPO', 'C4'),
    'PPOLag': ('PPO-Lag', 'C5'),
    'TRPOPID': ('TRPO-PID', 'C6'),
    'P3O': ('PPO-Lag', 'C7'),
    'FPO': ('FPO', 'C3'),
}

ENVTAGRANGES = {
    # 'SafetyPointGoal1-v0': {
    #     'cost': (-3, 30),
    #     'return': (-5, 30),
    # },
    # 'SafetyPointPush1-v0': {
    #     'cost': (-5, 50),
    #     'return': (-15, 40),
    # },
    # 'SafetyPointButton1-v0': {
    #     'cost': (-10, 100),
    #     'return': (-9, 20),
    # },
    # 'SafetyCarGoal1-v0': {
    #     'cost': (-5, 50),
    # },
    # 'SafetyCarPush1-v0': {
    #     'cost': (-5, 50),
    #     'return': (-5, 25),
    # },
    # 'SafetyCarButton1-v0': {
    #     'cost': (-15, 150),
    # },
    'SafetyCarCircle1-v0': {
        'cost': (-15, 150),
    },
    'SafetyAntVelocity-v1': {
        'cost': (-1, 10),
    },
    'SafetyHopperVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetySwimmerVelocity-v1': {
        'cost': (-15, 150),
    },
}

NOMAGNIFIERENVS = [
    'SafetyAntVelocity-v1',
    'SafetyHumanoidVelocity-v1',
]

COSTMAGNIFIERRANGES = {
    'SafetyPointGoal1-v0': (-0.2, 8),
    'SafetyPointPush1-v0': (-0.5, 5),
    'SafetyPointButton1-v0': (-0.5, 8),
    'SafetyCarGoal1-v0': (-0.1, 1),
    'SafetyCarPush1-v0': (-0.5, 5),
    'SafetyCarButton1-v0': (-1, 10),
    'SafetyPointCircle1-v0': (-1, 10),
    'SafetyCarCircle1-v0': (-1, 10),
    'SafetyHalfCheetahVelocity-v1': (-0.2, 2),
    'SafetyHopperVelocity-v1': (-0.2, 2),
    'SafetySwimmerVelocity-v1': (-0.2, 2),
    'SafetyWalker2dVelocity-v1': (-0.15, 1.5),
}

ENVTITLES = {
    'SafetyPointGoal1-v0': 'PointGoal',
    'SafetyPointPush1-v0': 'PointPush',
    'SafetyPointButton1-v0': 'PointButton',
    'SafetyCarGoal1-v0': 'CarGoal',
    'SafetyCarPush1-v0': 'CarPush',
    'SafetyCarButton1-v0': 'CarButton',
    'SafetyPointCircle1-v0': 'PointCircle',
    'SafetyCarCircle1-v0': 'CarCircle',
    'SafetyAntVelocity-v1': 'AntVelocity',
    'SafetyHumanoidVelocity-v1': 'HumanoidVelocity',
    'SafetyHalfCheetahVelocity-v1': 'HalfCheetahVelocity',
    'SafetyHopperVelocity-v1': 'HopperVelocity',
    'SafetySwimmerVelocity-v1': 'SwimmerVelocity',
    'SafetyWalker2dVelocity-v1': 'Walker2dVelocity',
}

ENVTAGLEFTMARGINS = {
    'SafetyPointCircle1-v0': {
        'cost': 0.15,
    },
    'SafetyCarCircle1-v0': {
        'cost': 0.15,
    },
    'SafetyAntVelocity-v1': {
        'return': 0.18,
    },
    'SafetyHalfCheetahVelocity-v1': {
        'return': 0.17,
    },
    'SafetyHopperVelocity-v1': {
        'return': 0.16,
    },
    'SafetyHumanoidVelocity-v1': {
        'return': 0.16,
    },
    'SafetySwimmerVelocity-v1': {
        'cost': 0.15,
        'return': 0.15,
    },
    'SafetyWalker2dVelocity-v1': {
        'return': 0.16,
    },
}


def epoch_to_step(epoch):
    return (epoch + 1) * 20000


def extract_training_data(
    envs: Sequence[str],
    algs: Sequence[str],
    tags: Sequence[str],
):
    for env in envs:
        for alg in algs:
            env_alg_dir = os.path.join(LOG_PATH, f'{alg}-' + '{' + env + '}')
            for log_dir_name in os.listdir(env_alg_dir):
                log = pd.read_csv(os.path.join(env_alg_dir, log_dir_name, 'progress.csv'))
                step = epoch_to_step(log['Train/Epoch'])
                seed = str(int(log_dir_name.split('-')[1]))
                for tag in tags:
                    df = pd.DataFrame(
                        {
                            'step': step,
                            'value': log[TAGLOGNAMES[tag]],
                        }
                    )
                    result_dir = os.path.join(RESULT_PATH, env, tag)
                    os.makedirs(result_dir, exist_ok=True)
                    result_file_name = '_'.join([alg, seed]) + '.csv'
                    result_file = os.path.join(result_dir, result_file_name)
                    df.to_csv(result_file, index=False)


def plot_training_curve(envs: Sequence[str], algs: Sequence[str], tags: Sequence[str], step: np.ndarray, magnify_last: float = 0.1):
    save_dir = os.path.join(FIGURE_PATH, 'training_curve')
    os.makedirs(save_dir, exist_ok=True)
    m = int(len(step) * magnify_last)
    for env in envs:
        for tag in tags:
            dfs = []
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                alg, seed = tag_file_name.split('.')[0].split('_')
                if alg not in algs:
                    continue
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                dfs.append(pd.DataFrame({
                    'step': step,
                    'value': np.interp(step, df['step'], df['value']),
                    'alg': alg,
                    'seed': seed,
                }))
            df = (
                pd.concat(dfs)
                .groupby(['alg', 'step'])
                .apply(mean_confidence_interval, include_groups=False)
                .reset_index()
            )
            algs = df['alg'].unique()
            algs = [alg for alg in ALGLABELCOLORS.keys() if alg in algs]  # sort
            sns.set_theme(style='dark')
            _, ax = plt.subplots(figsize=(5, 4))
            magnifier = tag == 'cost' and env not in NOMAGNIFIERENVS
            if magnifier:
                ax_inset = inset_axes(ax, width='40%', height='30%', loc='upper right')
                x1, x2 = step[-m], step[-1]
                y1, y2 = COSTMAGNIFIERRANGES[env]
            for alg in algs:
                df_alg = df[df['alg'] == alg]
                mean = df_alg['mean']
                ci = df_alg['ci']
                color = ALGLABELCOLORS[alg][1]
                ax.plot(step, mean, color=color)
                ax.fill_between(step, mean - ci, mean + ci, facecolor=color, alpha=0.2)
                if magnifier:
                    ax_inset.plot(step[-m:], mean[-m:], color=color)
                    ax_inset.fill_between(step[-m:], mean[-m:] - ci[-m:], mean[-m:] + ci[-m:],
                                          facecolor=color, alpha=0.2)
            ax.set_xlim(step[0], step[-1])
            ax.set_title(ENVTITLES[env])
            ax.set_xlabel('Environment step')
            ax.set_ylabel(TAGLABELS[tag])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if env in ENVTAGRANGES.keys() and tag in ENVTAGRANGES[env].keys():
                ax.set_ylim(ENVTAGRANGES[env][tag])
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.grid()
            if magnifier:
                ax_inset.set_xlim(x1, x2)
                ax_inset.set_ylim(y1, y2)
                ax_inset.grid()
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none',
                                zorder=10, clip_on=False)
                ax.add_patch(rect)
                con1 = ConnectionPatch(
                    xyA=(x1, y2), xyB=(0, 0),
                    coordsA='data', coordsB='axes fraction',
                    axesA=ax, axesB=ax_inset,
                    linestyle='--', color='black', alpha=0.5,
                    zorder=10, clip_on=False,
                )
                ax.add_artist(con1)
                con2 = ConnectionPatch(
                    xyA=(x2, y2), xyB=(1, 0),
                    coordsA='data', coordsB='axes fraction',
                    axesA=ax, axesB=ax_inset,
                    linestyle='--', color='black', alpha=0.5,
                    zorder=10, clip_on=False,
                )
                ax.add_artist(con2)
            if env in ENVTAGLEFTMARGINS.keys() and tag in ENVTAGLEFTMARGINS[env].keys():
                left = ENVTAGLEFTMARGINS[env][tag]
            else:
                left = 0.13
            plt.subplots_adjust(left=left, bottom=0.15, right=0.95, top=0.92)
            plt.savefig(os.path.join(save_dir, f'{env}_{tag}.{EXTENSION}'), dpi=300)
            plt.close()


def plot_legend():
    sns.set_theme(style='dark')
    plt.figure(figsize=(10, 1))
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=alg)
        for alg, color in ALGLABELCOLORS.values()
    ]
    plt.legend(handles=legend_elements, loc='center', ncol=len(legend_elements),
               handlelength=1, frameon=False)
    plt.axis('off')
    save_dir = os.path.join(FIGURE_PATH, 'training_curve')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'legend.{EXTENSION}'), dpi=300)
    plt.close()


def get_statistics(envs: Sequence[str], tags: Sequence[str], algs: Sequence[str], last: float = 0.1, window_length: int = 50):
    data = []
    for env in envs:
        for tag in tags:
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                alg, seed = tag_file_name.split('.')[0].split('_')
                if alg not in algs:
                    continue
                last_n = int(len(df) * last)
                mean_value = np.mean(df['value'][-last_n:])

                data.append({
                    'env': env,
                    'alg': alg,
                    'tag': tag,
                    'value': mean_value,
                    'seed': seed,
                })
                if tag == 'cost':
                    adjusted_area = _calculate_convergence_area(df['value'], mean_value, window_length)
                    data.append({
                        'env': env,
                        'alg': alg,
                        'tag': tag + '_adjusted_area',
                        'value': adjusted_area,
                        'seed': seed,
                    })
    keys = data[0].keys()
    data = {k: [d[k] for d in data] for k in keys}
    df = (
        pd.DataFrame(data)
        .groupby(['env', 'alg', 'tag'])
        .apply(mean_confidence_interval, include_groups=False)
        .reset_index()
    )
    os.makedirs(RESULT_PATH, exist_ok=True)
    df.to_csv(os.path.join(RESULT_PATH, 'statistics.csv'), float_format='%.2f', index=False)

def _calculate_convergence_area(values, mean_value, window_length):
    rolling_mean = np.array(values.rolling(window=window_length, min_periods=1).mean())
    
    max_index = np.argmax(rolling_mean)
    
    if max_index == len(rolling_mean) - 1:
        return np.inf

    remaining_values = rolling_mean[max_index:]

    remaining_indices = np.arange(len(remaining_values))
    target_indices = np.linspace(0, len(remaining_values) - 1, len(rolling_mean))

    interp_kind = 'cubic' if len(remaining_values) > 3 else 'linear'
    
    f = interpolate.interp1d(
        remaining_indices, 
        remaining_values,
        kind=interp_kind,
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    stretched_values = f(target_indices)
    
    adjusted_values = stretched_values - mean_value
    
    adjusted_area = np.sum(adjusted_values)
    
    if adjusted_area <= 0:
        return np.inf
        
    return adjusted_area

def plot_cost_return_scatter(envs: Sequence[str], algs: Sequence[str], normalize_by='PPO'):
    stats_file = os.path.join(RESULT_PATH, 'statistics.csv')
    if not os.path.exists(stats_file):
        print(f"Statistics file {stats_file} not found. Please run get_statistics() first.")
        return
    
    df = pd.read_csv(stats_file)
    df = df[df['env'].isin(envs) & df['alg'].isin(algs)]


    for env in envs:
        for tag in ['cost', 'return']:
            if not ((df['env'] == env) & (df['alg'] == normalize_by) & (df['tag'] == tag)).any():
                assert False, f"Missing {normalize_by} data for {env} in {tag}."
                
            baseline = df.loc[(df['env'] == env) & (df['alg'] == normalize_by) & (df['tag'] == tag), 'mean'].values[0]
            if baseline == 0:
                continue
                
            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] /= baseline
            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] = np.log(df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'])

    _, ax = plt.subplots(figsize=(10, 8))
    
    for alg in algs:
        if alg == normalize_by:
            continue
        ax.scatter(df.loc[(df['alg'] == alg) &(df['tag'] == 'cost'), 'mean'].mean(),  
                    df.loc[(df['alg'] == alg) &(df['tag'] == 'return'), 'mean'].mean(),
                    marker='*', 
                    s=200, 
                    label=alg)
    ax.set_xlim(ax.get_xlim()[::-1])

    plt.xlabel(f'Normalized log cost', fontsize=12)
    plt.ylabel(f'Normalized log return', fontsize=12)
    plt.title(f'Algorithm Performance Comparison(relative to {normalize_by})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best',fontsize=10)

    plt.savefig(os.path.join(RESULT_PATH, f'return_cost_scatter_log.{EXTENSION}'), dpi=300)

def mean_confidence_interval(group, include_groups=False):
    mean = group['value'].mean()
    std = group['value'].std()
    n = group['seed'].nunique()
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0  # 0.95 confidence interval
    return pd.Series({'mean': mean, 'ci': ci})

def plot_cost_convergence_bargraph(envs: Sequence[str], algs: Sequence[str]):
    stats_file = os.path.join(RESULT_PATH, 'statistics.csv')
    if not os.path.exists(stats_file):
        print(f"Statistics file {stats_file} not found. Please run get_statistics() first.")
        return
    
    df = pd.read_csv(stats_file)
    df = df[df['env'].isin(envs) & df['alg'].isin(algs)]
    
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(14, 8))
    
    tag_type = 'cost_adjusted_area'
    tag_data = df[df['tag'] == tag_type].copy()
    
    finite_max = tag_data[tag_data['mean'] != np.inf]['mean'].max() if any(tag_data['mean'] != np.inf) else 1000
    
    tag_data.loc[tag_data['mean'] == np.inf, 'mean'] = finite_max * 1.2
    
    ax = sns.barplot(x='env', y='mean', hue='alg', data=tag_data)
    
    plt.axhline(y=finite_max * 1.2, color='red', linestyle='--', alpha=0.7)
    
    x_max = len(tag_data['env'].unique()) - 1 + 0.5  
    plt.text(x_max, finite_max * 1.2, '∞', color='red', ha='right', va='bottom', fontsize=16)
    
    plt.title('Cost Convergence Area by Environment and Algorithm', fontsize=16)
    plt.xlabel('Environment', fontsize=14)
    plt.ylabel('Adjusted Area (larger is worse)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    import matplotlib.ticker as ticker
    def thousands(x, pos):
        return f'{x/1000:.1f}K'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(thousands))
    
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, f'{tag_type}_bargraph.{EXTENSION}'), dpi=300)
    plt.close()
   
def plot_cost_observation(
        data_path, 
        x_range: tuple = (-2, 2),
        y_range: tuple = (-2, 2),
        save_path: str = f'heatmap.{EXTENSION}',
    ):
    data = np.load(data_path)
    values_c = data['values_c']
    plt.figure(figsize=(8, 6))
    plt.imshow(values_c, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower')
    plt.colorbar(label='Critic Value')
    plt.title('Critic Value Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(save_path)
