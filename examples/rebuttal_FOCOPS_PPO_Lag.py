import os
from omnisafe.utils.path import RESULT_PATH
from omnisafe.utils.visualizer import extract_training_data, get_statistics, ENVTITLES
import pandas as pd

envs = [
    "SafetyCarGoal1-v0",
    "SafetyCarPush1-v0",
    # "SafetyCarButton1-v0",
    # "SafetyPointGoal1-v0",
    # "SafetyPointPush1-v0",
    "SafetyPointButton1-v0",
]

base_alg = "FPO"
compare_algs = ["FOCOPS", "PPOLag"]
tags = ["cost", "return"]
algs = ["FPO", "FOCOPS", "PPOLag", "PPO"]
df = get_statistics(envs, tags, algs)

def get_mean(env, alg, tag):
    sub = df[(df["env"] == env) & (df["alg"] == alg) & (df["tag"] == tag)]
    return sub["mean"].iloc[0]


markdown_lines = []

header1 = [
    "Environment",
    "FPO", "",                          
    "FOCOPS", "", "", "", "", "",      
    "PPOLag", "", "", "", "", "",      
]
markdown_lines.append("|" + "|".join(header1) + "|")

separator = ["-"] * len(header1)
markdown_lines.append("|" + "|".join(separator) + "|")

header2 = [
    "",           
    "Cost", "Return",                        
    "Cost", "Return", "Cost Δ", "Return Δ", "Cost Δ%", "Return Δ%",   
    "Cost", "Return", "Cost Δ", "Return Δ", "Cost Δ%", "Return Δ%",   
]
markdown_lines.append("|" + "|".join(header2) + "|")


sum_dict = {
    "FPO_cost": 0.0,
    "FPO_return": 0.0,
    "FOCOPS_cost": 0.0,
    "FOCOPS_return": 0.0,
    "FOCOPS_cost_delta": 0.0,
    "FOCOPS_return_delta": 0.0,
    "FOCOPS_cost_pct": 0.0,
    "FOCOPS_return_pct": 0.0,
    "PPOLag_cost": 0.0,
    "PPOLag_return": 0.0,
    "PPOLag_cost_delta": 0.0,
    "PPOLag_return_delta": 0.0,
    "PPOLag_cost_pct": 0.0,
    "PPOLag_return_pct": 0.0,
}

n_env = len(envs)


# normalize by PPO
for env in envs:
    ppo_cost = get_mean(env, "PPO", "cost")
    ppo_ret = get_mean(env, "PPO", "return")

    fpo_cost = get_mean(env, base_alg, "cost") / ppo_cost
    fpo_return = get_mean(env, base_alg, "return") / ppo_ret

    foc_cost = get_mean(env, "FOCOPS", "cost") / ppo_cost
    foc_ret = get_mean(env, "FOCOPS", "return") / ppo_ret
    foc_cost_delta = foc_cost - fpo_cost
    foc_ret_delta = foc_ret - fpo_return
    foc_cost_pct = foc_cost_delta / fpo_cost * 100 if fpo_cost != 0 else float("nan")
    foc_ret_pct = foc_ret_delta / fpo_return * 100 if fpo_return != 0 else float("nan")

    ppo_lag_cost = get_mean(env, "PPOLag", "cost") / ppo_cost
    ppo_lag_ret = get_mean(env, "PPOLag", "return") / ppo_ret
    ppo_lag_cost_delta = ppo_lag_cost - fpo_cost
    ppo_lag_ret_delta = ppo_lag_ret - fpo_return
    ppo_lag_cost_pct = ppo_lag_cost_delta / fpo_cost * 100 if fpo_cost != 0 else float("nan")
    ppo_lag_ret_pct = ppo_lag_ret_delta / fpo_return * 100 if fpo_return != 0 else float("nan")

    row = [
        env,
        f"{fpo_cost:.4f}", f"{fpo_return:.4f}",
        f"{foc_cost:.4f}", f"{foc_ret:.4f}", f"{foc_cost_delta:.4f}", f"{foc_ret_delta:.4f}",
        f"{foc_cost_pct:.4f}", f"{foc_ret_pct:.4f}",
        f"{ppo_lag_cost:.4f}", f"{ppo_lag_ret:.4f}", f"{ppo_lag_cost_delta:.4f}", f"{ppo_lag_ret_delta:.4f}",
        f"{ppo_lag_cost_pct:.4f}", f"{ppo_lag_ret_pct:.4f}",
    ]
    markdown_lines.append("|" + "|".join(row) + "|")

    sum_dict["FPO_cost"] += fpo_cost
    sum_dict["FPO_return"] += fpo_return

    sum_dict["FOCOPS_cost"] += foc_cost
    sum_dict["FOCOPS_return"] += foc_ret
    sum_dict["FOCOPS_cost_delta"] += foc_cost_delta
    sum_dict["FOCOPS_return_delta"] += foc_ret_delta
    sum_dict["FOCOPS_cost_pct"] += foc_cost_pct
    sum_dict["FOCOPS_return_pct"] += foc_ret_pct

    sum_dict["PPOLag_cost"] += ppo_lag_cost
    sum_dict["PPOLag_return"] += ppo_lag_ret
    sum_dict["PPOLag_cost_delta"] += ppo_lag_cost_delta
    sum_dict["PPOLag_return_delta"] += ppo_lag_ret_delta
    sum_dict["PPOLag_cost_pct"] += ppo_lag_cost_pct
    sum_dict["PPOLag_return_pct"] += ppo_lag_ret_pct

avg_row = ["**Average**"]

def avg(key):
    return sum_dict[key] / n_env

avg_row.extend([
    f"{avg('FPO_cost'):.4f}", f"{avg('FPO_return'):.4f}",
    f"{avg('FOCOPS_cost'):.4f}", f"{avg('FOCOPS_return'):.4f}",
    f"{avg('FOCOPS_cost_delta'):.4f}", f"{avg('FOCOPS_return_delta'):.4f}",
    f"{avg('FOCOPS_cost_pct'):.4f}", f"{avg('FOCOPS_return_pct'):.4f}",
    f"{avg('PPOLag_cost'):.4f}", f"{avg('PPOLag_return'):.4f}",
    f"{avg('PPOLag_cost_delta'):.4f}", f"{avg('PPOLag_return_delta'):.4f}",
    f"{avg('PPOLag_cost_pct'):.4f}", f"{avg('PPOLag_return_pct'):.4f}",
])

markdown_lines.append("|" + "|".join(avg_row) + "|")

table_str = "\n".join(markdown_lines)
print(table_str)

out_path = os.path.join(RESULT_PATH, "FOCOPS_PPO_Lag_compare.md")
with open(out_path, "w") as f:
    f.write(table_str)
