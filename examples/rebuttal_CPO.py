import pandas as pd

def count_infeasible(csv_path):
    df = pd.read_csv(csv_path)

    infeasible_num_values = df['Misc/infeasible_num'].head(500)
    cost_values = df['Metrics/EpCost'].head(500)
    ret_values = df['Metrics/EpRet'].head(500)


    # 打印 Markdown 表头
    print("| Epoch Range     | infeasible_num | cost | return |")
    print("|-|-|-|-|")

    # 每100个为一组统计
    for i in range(0, 500, 100):
        start = i
        end = min(i + 100, len(infeasible_num_values))
        if start == 0:
            infeasible_group_sum = int(infeasible_num_values[start:end].sum())
        else:
            infeasible_group_sum = int(infeasible_num_values[start:end].sum() - infeasible_num_values[start-100:end-100].sum())
        cost_group_sum = int(cost_values[start:end].sum())
        ret_group_sum = int(ret_values[start:end].sum())
        print(f"| epoch: {start}–{end} | {infeasible_group_sum} | {cost_group_sum} | {ret_group_sum} |")

if __name__ == "__main__":
    # 将下面路径替换为你的CSV文件路径
    csv_file_path = "/home/sunyuanxu/FPO-Algorithm/examples/runs/CPO-{SafetyPointGoal1-v0}/seed-000-2025-07-30-09-34-38-infeasible/progress.csv"
    count_infeasible(csv_file_path)
