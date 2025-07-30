import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
from omnisafe.utils.path import LOG_PATH, RESULT_PATH, FIGURE_PATH


EXTENSION = 'pdf'

TAGLOGNAMES = {
    'cost': 'Metrics/EpCost',
    'return': 'Metrics/EpRet',
    'error_0': 'Freq/value_c_lt_0.1_0',
    'error_20': 'Freq/value_c_lt_0.1_20',
    'error_40': 'Freq/value_c_lt_0.1_40',
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
    'CPPOPID': ('CPPO-PID', 'C8'),
    'P3O': ('P3O', 'C7'),
    'FPO': ('FPO', 'C3'),
}

ALGLABELS = {
    'CPO': 'CPO',
    'PCPO': 'PCPO',
    'FOCOPS': 'FOCOPS',
    'RCPO': 'RCPO',
    'PPOLag': 'PPO-Lag',
    'TRPOPID': 'TRPO-PID',
    'P3O': 'P3O',
    'FPO': 'FPO',
}

ENVTAGRANGES = {
    'SafetyPointGoal1-v0': {
        'cost': (-8, 80),
    },
    'SafetyPointPush1-v0': {
        'cost': (-8, 80),
    },
    'SafetyPointButton1-v0': {
        'cost': (-13, 130),
    },
    'SafetyCarGoal1-v0': {
        'cost': (-4, 40),
    },
    'SafetyCarPush1-v0': {
        'cost': (-6, 60),
    },
    'SafetyPointCircle1-v0': {
        'cost': (-25, 250),
    },
    'SafetyCarCircle1-v0': {
        'cost': (-15, 150),
    },
    'SafetyAntVelocity-v1': {
        'cost': (-1, 10),
    },
    'SafetyHalfCheetahVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetyHopperVelocity-v1': {
        'cost': (-5, 50),
    },
    'SafetySwimmerVelocity-v1': {
        'cost': (-9, 90),
    },
}

NOMAGNIFIERENVS = [
    'SafetyAntVelocity-v1',
    'SafetyHumanoidVelocity-v1',
]

COSTMAGNIFIERRANGES = {
    'SafetyPointGoal1-v0': (-0.6, 6),
    'SafetyPointPush1-v0': (-0.6, 6),
    'SafetyPointButton1-v0': (-0.2, 12),
    'SafetyCarGoal1-v0': (-0.2, 2),
    'SafetyCarPush1-v0': (-0.5, 5),
    'SafetyCarButton1-v0': (-1.5, 15),
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
    'SafetyPointButton1-v0': {
        'cost': 0.15,
    },
    'SafetyCarButton1-v0': {
        'cost': 0.15,
    },
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

def extract_training_data_with_type(
    envs: Sequence[str],
    algs: Sequence[str],
    tags: Sequence[str],
    alg_type: Optional[str] = None,
):
    for env in envs:
        for alg in algs:
            env_alg_dir = os.path.join(LOG_PATH, f'{alg}-' + '{' + env + '}')
            for log_dir_name in os.listdir(env_alg_dir):
                log = pd.read_csv(os.path.join(env_alg_dir, log_dir_name, 'progress.csv'))
                step = epoch_to_step(log['Train/Epoch'])
                
                # Parse log_dir_name to extract seed and possible type
                parts = log_dir_name.split('-')
                seed = str(int(parts[1]))  # Extract seed number
                
                # Check if there's a type suffix (anything after the timestamp)
                # Format: seed-XXX-YYYY-MM-DD-HH-MM-SS[-type]
                if alg_type is not None and '-'.join(parts[8:]) != alg_type:
                    continue

                if len(parts) > 8:  # More than 6 parts means there's a type suffix
                    type_suffix = '-'.join(parts[8:])  # Join all parts after timestamp
                    effective_alg = f"{alg}-{type_suffix}"
                else:
                    effective_alg = alg
                
                for tag in tags:
                    df = pd.DataFrame(
                        {
                            'step': step,
                            'value': log[TAGLOGNAMES[tag]],
                        }
                    )
                    result_dir = os.path.join(RESULT_PATH, env, tag)
                    os.makedirs(result_dir, exist_ok=True)
                    result_file_name = '_'.join([effective_alg, seed]) + '.csv'
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


def get_statistics(envs: Sequence[str], tags: Sequence[str], algs: Sequence[str], last: float = 0.1):
    data = []
    for env in envs:
        for tag in tags:
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                alg, seed = tag_file_name.rsplit('.', 1)[0].split('_', 1)
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
    return df

def get_statistics_with_type(envs: Sequence[str], tags: Sequence[str], algs: Sequence[str], last: float = 0.1):
    data = []
    for env in envs:
        for tag in tags:
            tag_dir = os.path.join(RESULT_PATH, env, tag)
            for tag_file_name in os.listdir(tag_dir):
                tag_file = os.path.join(tag_dir, tag_file_name)
                df = pd.read_csv(tag_file)
                
                # Parse filename to extract algorithm and seed
                file_base = tag_file_name[:-4]  # Remove .csv extension
                parts = file_base.split('_')
                seed = parts[-1]  # Last part is always seed
                alg = '_'.join(parts[:-1])  # Everything except the last part is algorithm
                
                if alg not in algs:
                    continue
                
                last_n = int(len(df) * last)
                mean_value = np.mean(df['value'][-last_n:])

                data.append({
                    'env': env,
                    'alg': alg,  # This now includes type suffix if present
                    'tag': tag,
                    'value': mean_value,
                    'seed': seed,
                })
    
    if not data:
        return pd.DataFrame()
        
    keys = data[0].keys()
    data = {k: [d[k] for d in data] for k in keys}
    df = (
        pd.DataFrame(data)
        .groupby(['env', 'alg', 'tag'])  # Group by full algorithm name (including type)
        .apply(mean_confidence_interval, include_groups=False)
        .reset_index()
    )
    os.makedirs(RESULT_PATH, exist_ok=True)
    df.to_csv(os.path.join(RESULT_PATH, 'statistics_type.csv'), float_format='%.2f', index=False)
    return df

def get_table():
    df = pd.read_csv(os.path.join(RESULT_PATH, 'statistics.csv'))
    envs = sorted(df['env'].unique())
    algs = ALGLABELS.keys()

    pivot_df = df.pivot_table(index=['alg', 'env'], columns='tag', values=['mean', 'ci'])

    pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    latex_table = r'''\begin{table}[ht]
    \centering
    \caption{Average cost and return in the last 10\% iterations}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccc}
        \toprule
'''
    tab = '    '

    for i in range(0, len(envs), 3):
        group_envs = envs[i:min(i + 3, len(envs))]

        latex_table += tab * 2
        for env in group_envs:
            latex_table += f' & \multicolumn{{2}}{{c}}{{{ENVTITLES[env]}}}'
        latex_table += ' \\\\\n'

        latex_table += tab * 2
        for j in range(len(group_envs)):
            latex_table += f'\cmidrule(lr){{{j * 2 + 2}-{j * 2 + 3}}} '
        latex_table += '\n'

        latex_table += tab * 2 + 'Algorithm'
        latex_table += ' & Cost & Return' * len(group_envs)
        latex_table += ' \\\\\n'
        latex_table += tab * 2 + '\midrule\n'

        for alg in algs:
            latex_table += tab * 2 + ALGLABELS[alg]
            for env in group_envs:
                cost_row = pivot_df[(pivot_df['alg'] == alg) & (pivot_df['env'] == env)]
                cost_mean = cost_row['cost_mean'].values[0]
                cost_ci = cost_row['cost_ci'].values[0]
                cost_str = f'${cost_mean:.2f}\\pm{cost_ci:.2f}$'

                ret_row = pivot_df[(pivot_df['alg'] == alg) & (pivot_df['env'] == env)]
                ret_mean = ret_row['return_mean'].values[0]
                ret_ci = ret_row['return_ci'].values[0]
                ret_str = f'${ret_mean:.2f}\\pm{ret_ci:.2f}$'

                latex_table += f' & {cost_str} & {ret_str}'
            latex_table += ' \\\\\n'

        if i + 3 < len(envs):
            latex_table += tab * 2 + '\midrule\n'

    latex_table += r'''        \bottomrule
    \end{tabular}
    }
\end{table}'''

    with open('table.tex', 'w') as f:
        f.write(latex_table)

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
                assert False, f"Baseline value for {normalize_by} in {env} is zero."
                
            df.loc[(df['env'] == env) & (df['tag'] == tag), 'mean'] /= baseline

    sns.set_theme(style='dark')
    _, ax = plt.subplots(figsize=(6, 5))
    for alg in algs:
        if alg == normalize_by:
            continue
        label, color = ALGLABELCOLORS[alg]
        cost = df.loc[(df['alg'] == alg) & (df['tag'] == 'cost'), 'mean']
        ret = df.loc[(df['alg'] == alg) & (df['tag'] == 'return'), 'mean']
        cost_mean = cost.mean()
        ret_mean = ret.mean()
        cost_ci = 1.96 * cost.std() / len(cost)
        ret_ci = 1.96 * ret.std() / len(ret)
        ax.errorbar(
            cost_mean,
            ret_mean,
            xerr=cost_ci,
            yerr=ret_ci,
            label=label,
            color=color,
            fmt='o',
            markersize=6,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
    ax.set_xlim(ax.get_xlim()[1], 0)
    ax.set_ylim(ax.get_ylim()[0], 0.8)
    ax.set_xlabel('Normalized cost')#, fontsize=12)
    ax.set_ylabel('Normalized return')#, fontsize=12)
    plt.grid()#True, linestyle='--', alpha=1, linewidth=1.5)
    plt.legend(loc='lower left')#, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f'return_cost_scatter.{EXTENSION}'), dpi=300)

def mean_confidence_interval(group, include_groups=False):
    mean = group['value'].mean()
    std = group['value'].std()
    n = group['seed'].nunique()
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0  # 0.95 confidence interval
    return pd.Series({'mean': mean, 'ci': ci})

def plot_cost_observation(
        data_dir: str, 
        x_range: tuple = (-2, 2),
        y_range: tuple = (-2, 2),
        epoch_list: Sequence = [5, 10, 20, 50, 100],
        seed: int = 42,
        # figsize_single: tuple = (8, 6),
        figsize_combined: tuple = (24, 5),
    ):
    epoch_data = {}
    
    global_max = float('-inf')
    
    for epoch in epoch_list:
        try:
            data_path = os.path.join(data_dir, f'saved_obs_epoch-{epoch}_{seed}.npz')
            data = np.load(data_path)
            values_c = data['values_c']
            
            epoch_data[epoch] = values_c 
            
            # global_min = min(global_min, values_c.min())
            global_max = max(global_max, values_c.max())
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Unable to load data for Epoch {epoch}: {e}")
    
    if not epoch_data:
        print("No valid data to plot")
        return
    global_min = 0.2 - global_max
    
    if len(epoch_data) > 1:
        rows = 1
        cols = len(epoch_data) 
        
        fig, axs = plt.subplots(rows, cols, figsize=figsize_combined)
        
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        
        dummy_data = np.array([[global_min, global_max], [global_min, global_max]])
        
        for idx, (epoch, values_c) in enumerate(sorted(epoch_data.items())):
            if idx >= cols:
                print(f"Warning: Can only display the first {cols} epochs")
                break
                
            ax = axs[idx] if len(axs.shape) == 1 else axs[0, idx]
            
            im = ax.imshow(values_c, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
                          origin='lower', cmap='coolwarm', vmin=global_min, vmax=global_max)
                
            ax.contour(
                np.linspace(x_range[0], x_range[1], values_c.shape[1]),
                np.linspace(y_range[0], y_range[1], values_c.shape[0]),
                values_c,
                levels=[0.1],
                colors='black',
                linewidths=1.5
            )
            # ax.clabel(contour, inline=True, fontsize=10, fmt='%.1f')
            
            ax.set_title(f'Epoch {epoch}', fontsize=14)
            ax.set_xlabel('X Position', fontsize=12)
            if idx == 0: 
                ax.set_ylabel('Y Position', fontsize=12)
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_aspect('equal')
            ax.grid(False)
            
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        dummy_ax = fig.add_axes([0, 0, 0, 0]) 
        dummy_im = dummy_ax.imshow(dummy_data, cmap='coolwarm', vmin=global_min, vmax=global_max)
        dummy_ax.set_visible(False)
        
        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7]) 
        cbar = fig.colorbar(dummy_im, cax=cbar_ax)
        cbar.set_label('Critic Value', fontsize=12)
        cbar.ax.tick_params(labelsize=10) 
            
        plt.savefig(os.path.join(data_dir, f'heatmap_all_epochs_{seed}.{EXTENSION}'), dpi=300)
        plt.close()
        
    return len(epoch_data)
