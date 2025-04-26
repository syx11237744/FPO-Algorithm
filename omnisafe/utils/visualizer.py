import os
from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
from omnisafe.utils.path import LOG_PATH, RESULT_PATH, FIGURE_PATH


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
    'RCPO': ('RCPO', 'C1'),
    'FOCOPS': ('FOCOPS', 'C2'),
    'PPOLag': ('PPO-Lag', 'C4'),
    'P3O': ('PPO-Lag', 'C5'),
    'PCPO': ('PCPO', 'C6'),
    'TRPOPID': ('TRPO-PID', 'C7'),
    'FPO': ('FPO', 'C3'),
}

ENVTAGRANGES = {
    'SafetyPointGoal1-v0': {
        'cost': (-3, 30),
        'return': (-5, 30),
    },
    'SafetyPointPush1-v0': {
        'cost': (-5, 50),
        'return': (-15, 40),
    },
    'SafetyPointButton1-v0': {
        'cost': (-10, 100),
        'return': (-9, 9),
    },
    'SafetyPointCircle1-v0': {
        'cost': (-5, 50),
    },
    'SafetyCarGoal1-v0': {
        'cost': (-5, 50),
    },
    'SafetyCarPush1-v0': {
        'cost': (-5, 50),
        'return': (-5, 25),
    },
    'SafetyCarButton1-v0': {
        'cost': (-15, 150),
    },
    'SafetyCarCircle1-v0': {
        'cost': (-5, 50),
    },
    'SafetyAntVelocity-v1': {
        'cost': (-0.5, 5),
    },
    'SafetyHalfCheetahVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetyHopperVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetyHumanoidVelocity-v1': {
        'cost': (-0.1, 1),
    },
    'SafetyWalker2dVelocity-v1': {
        'cost': (-1, 10),
    },
    'SafetySwimmerVelocity-v1': {
        'cost': (-10, 100),
    },
}

COSTMAGNIFIERRANGES = {
    'SafetyPointGoal1-v0': (-0.2, 2),
    'SafetyPointPush1-v0': (-0.5, 5),
    'SafetyCarGoal1-v0': (-0.1, 1),
    'SafetyCarPush1-v0': (-0.5, 5),
    'SafetyPointCircle1-v0': (-0.1, 1),
    'SafetyCarCircle1-v0': (-0.1, 1),
    'SafetyAntVelocity-v1': (-0.05, 0.5),
    'SafetyPointButton1-v0': (-0.5, 5),
    'SafetyCarButton1-v0': (-1, 10),
    'SafetyHalfCheetahVelocity-v1': (-0.2, 2),
    'SafetyHopperVelocity-v1': (-0.1, 1),
    'SafetyWalker2dVelocity-v1': (-0.05, 0.5),
    'SafetySwimmerVelocity-v1': (-0.1, 1),
}

ENVTITLES = {
    'SafetyPointGoal1-v0': 'PointGoal',
    'SafetyPointPush1-v0': 'PointPush',
    'SafetyPointCircle1-v0': 'PointCircle',
    'SafetyCarGoal1-v0': 'CarGoal',
    'SafetyCarPush1-v0': 'CarPush',
    'SafetyCarCircle1-v0': 'CarCircle',
    'SafetyAntVelocity-v1': 'AntVelocity',
    'SafetyHumanoidVelocity-v1': 'HumanoidVelocity',
    'SafetyPointButton1-v0': 'PointButton',
    'SafetyCarButton1-v0': 'CarButton',
    'SafetyHalfCheetahVelocity-v1': 'HalfCheetahVelocity',
    'SafetyHopperVelocity-v1': 'HopperVelocity',
    'SafetyWalker2dVelocity-v1': 'Walker2dVelocity',
    'SafetySwimmerVelocity-v1': 'SwimmerVelocity',
}

NOMAGNIFIERENVS = [
    'SafetyHumanoidVelocity-v1',
]


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


def plot_training_curve(envs: Sequence[str], tags: Sequence[str], step: np.ndarray, magnify_last: float = 0.1):
    save_dir = os.path.join(FIGURE_PATH, 'training_curve')
    os.makedirs(save_dir, exist_ok=True)
    m = int(len(step) * magnify_last)
    for env in envs:
        for tag in tags:
            dfs = []
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                alg, seed = tag_file_name.split('.')[0].split('_')
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
            plt.subplots_adjust(
                left=0.13,    # Increase left margin (for y-axis label)
                bottom=0.15,  # Increase bottom margin (for x-axis label)
                right=0.95,   # Decrease right margin (for inset space)
                top=0.92,     # Decrease top margin (for title/inset)
            )
            plt.savefig(os.path.join(save_dir, f'{env}_{tag}.png'), dpi=300)
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
    plt.savefig(os.path.join(save_dir, 'legend.png'), dpi=300)
    plt.close()


def get_statistics(envs: Sequence[str], tags: Sequence[str], last: float = 0.1, window_length: int = 10):
    data = []
    convergence_data = []
    for env in envs:
        for tag in tags:
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            # import pdb; pdb.set_trace()
            for tag_file_name in os.listdir(tag_dir):
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                alg, seed = tag_file_name.split('.')[0].split('_')
                last_n = int(len(df) * last)
                mean_value = np.mean(df['value'][-last_n:])

                data.append({
                    'env': env,
                    'alg': alg,
                    'tag': tag,
                    'value': mean_value,
                    'seed': seed,
                })
                if tag in ['cost', 'return']:
                    # calculate the first step to reach the final cost level
                    threshold_step = None
                    rolling_mean = df['value'].rolling(window=window_length, min_periods=1).mean()
                    for i, row in reversed(list(df.iterrows())):
                        if ((tag == 'return' and rolling_mean[i] <= mean_value * 0.8) or 
                            (tag == 'cost' and rolling_mean[i] >= mean_value * 1.2)):
                            threshold_step = row['step']
                            break
                    convergence_data.append({
                        'env': env,
                        'alg': alg,
                        'tag': tag + '_threshold_step',
                        'value': threshold_step,
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

    keys = convergence_data[0].keys()
    convergence_data = {k: [d[k] for d in convergence_data] for k in keys}
    df = (
        pd.DataFrame(convergence_data)
        .groupby(['env', 'alg', 'tag'])
        .apply(mean_confidence_interval, include_groups=False)
        .reset_index()
    )
    # drop the 'ci' column
    df = df.drop(columns=['ci'])
    df.to_csv(os.path.join(RESULT_PATH, 'convergence_statistics.csv'), float_format='%.2f', index=False)


def draw_cost_return_scatter(stats_file, normalize_by='PPO'):
    df = pd.read_csv(stats_file)
    
    algorithms = df['alg'].unique()
    environments = df['env'].unique()
    tags = df['tag'].unique()

    for env in environments:
        for tag in tags:
            # normalize by "normalize_by"
            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] /= df.loc[
                (df['env'] == env) & (df['alg'] == normalize_by) & (df['tag'] == tag), 'mean'].values[0]
    df.to_csv("temp.csv", index=False)
    # cost_sums = {alg: 0 for alg in algorithms}
    # return_sums = {alg: 0 for alg in algorithms}
    
    # for alg in algorithms:
    #     for env in environments:
    #         return_row = df[(df['env'] == env) & (df['alg'] == alg) & (df['tag'] == 'return')]
    #         if not return_row.empty:
    #             return_sums[alg] += return_row['mean'].values[0]
            
    #         cost_row = df[(df['env'] == env) & (df['alg'] == alg) & (df['tag'] == 'cost')]
    #         if not cost_row.empty:
    #             cost_sums[alg] += cost_row['mean'].values[0]
    
    # if normalize_by in cost_sums and normalize_by in return_sums:
    #     ppo_cost = cost_sums[normalize_by]
    #     ppo_return = return_sums[normalize_by]
        
    #     for alg in algorithms:
    #         cost_sums[alg] /= ppo_cost
    #         return_sums[alg] /= ppo_return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for alg in algorithms:
        ax.scatter(df.loc[(df['alg'] == alg) &(df['tag'] == 'cost'), 'mean'].mean(),  
                    df.loc[(df['alg'] == alg) &(df['tag'] == 'return'), 'mean'].mean(),
                    marker='*', 
                    s=200, 
                    label=alg)
    
    # max_cost = max(cost_sums.values()) * 1.1
    # max_return = max(return_sums.values()) * 1.1
    # plt.xlim(max_cost, 0)
    # plt.ylim(0, max_return)
    # reverse x axis
    ax.set_xlim(ax.get_xlim()[::-1])   
    
    plt.xlabel('Cost (normalized)', fontsize=12)
    plt.ylabel('Return (normalized)', fontsize=12)
    plt.title('Algorithm Performance: Return vs Cost', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(RESULT_PATH, 'return_cost_scatter.png'), dpi=300)

def mean_confidence_interval(group, include_groups=False):
    mean = group['value'].mean()
    std = group['value'].std()
    n = group['seed'].nunique()
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0  # 0.95 confidence interval
    return pd.Series({'mean': mean, 'ci': ci})


def plot_threshold_bargraph(csv_file):
    df = pd.read_csv(csv_file)

    sns.set_theme(style='whitegrid')
    

    for tag_type in ['cost_threshold_step', 'return_threshold_step']:
        plt.figure(figsize=(14, 8))
        
        tag_data = df[df['tag'] == tag_type]
        
        ax = sns.barplot(x='env', y='mean', hue='alg', data=tag_data)
        
        metric_name = 'Cost' if 'cost' in tag_type else 'Return'
        plt.title(f'First step to reach final {metric_name} level by environment and algorithm', fontsize=16)
        plt.xlabel('Environment', fontsize=14)
        plt.ylabel('Environment Steps', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        
        import matplotlib.ticker as ticker
        def millions(x, pos):
            return f'{x/1e6:.1f}M'
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
        
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULT_PATH, f'{tag_type}_bargraph.png'), dpi=300)
        plt.close()
    
    plt.figure(figsize=(18, 10))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    cost_data = df[df['tag'] == 'cost_threshold_step']
    sns.barplot(x='env', y='mean', hue='alg', data=cost_data, ax=ax1)
    ax1.set_title('First step to reach final Cost level', fontsize=16)
    ax1.set_xlabel('Environment', fontsize=14)
    ax1.set_ylabel('Environment Steps', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    
    return_data = df[df['tag'] == 'return_threshold_step']
    sns.barplot(x='env', y='mean', hue='alg', data=return_data, ax=ax2)
    ax2.set_title('First step to reach final Return level', fontsize=16)
    ax2.set_xlabel('Environment', fontsize=14)
    ax2.set_ylabel('Environment Steps', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(millions))
    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.get_legend().remove()
    fig.legend(handles, labels, title='Algorithm', bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULT_PATH, 'combined_threshold_bargraph.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
# plot_threshold_bargraph('threshold_statistics.csv', 'figures')