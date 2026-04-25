import torch

# 加载模型权重
# pruned_model_path = "output/glue/mrpc/bert-large-uncased/0.17_-0.5/temp/models/compressed_model.pth"  # 剪枝后模型路径
# original_model_path = "output/glue/mrpc/bert-large-uncased/0.17_-0.5/temp/models/init_model.pth"  # 剪枝前模型路径



def calculate_compression_rate(original_model, pruned_model):
    # 原始模型的参数量
    total_params_before = sum(p.numel() for p in original_model.parameters())
    # 剪枝后模型的参数量
    total_params_after = sum((p != 0).sum().item() for p in pruned_model.parameters())

    # 压缩率计算
    compression_rate = 1 - (total_params_after / total_params_before)
    print(f"Total Params Before Pruning: {total_params_before}")
    print(f"Total Params After Pruning: {total_params_after}")
    print(f"Compression Rate: {compression_rate * 100:.2f}%")
    return compression_rate

# pruned_model = torch.load(pruned_model_path)
# original_model = torch.load(original_model_path)
# calculate_compression_rate(original_model, pruned_model)

# import pickle
# import numpy as np
# # 加载保存的分析日志
# with open("all_analysis_logs.pkl", "rb") as f:
#     analysis_logs = pickle.load(f)

# # 提取 delta_W 和 grad
# delta_W_logs = analysis_logs["delta_W"]['bert.encoder.layer.11.output.dense.weight']
# grad_logs = analysis_logs["grad"]['bert.encoder.layer.11.output.dense.weight']
# # delta_W_norms = {k: np.linalg.norm(v, ord="fro") for k, v in delta_W_logs.items()}
# # grad_norms = {k: np.linalg.norm(v, ord="fro") for k, v in grad_logs.items()}
# import torch

# # 假设 delta_W_logs 是一个 PyTorch 张量
# column_sums = np.sqrt(delta_W_logs).sum(axis=0)  # 按列求和
# sorted_indices = np.argsort(column_sums)  # 按和排序，返回排序后的索引

# column_sums = grad_logs.sum(axis=0)  # 按列求和
# grad_sorted_indices = np.argsort(column_sums)
# # 输出排序后的列索引
# print("排序后的列索引：", sorted_indices[:128].tolist())  
# print("排序后的列索引：", grad_sorted_indices[:128].tolist()) # 转为 Python 列表输出
# print("差值：",sorted_indices[2550:]-grad_sorted_indices[2550:])
# print(f'delta_W_logs:{delta_W_logs.sum(dim=0)},grad_logs:{grad_logs.sum()}')

# for key in delta_W_logs.keys():
#     delta_mean = np.mean(delta_W_logs[key])
#     grad_mean = np.mean(grad_logs[key])
#     delta_std = np.std(delta_W_logs[key])
#     grad_std = np.std(grad_logs[key])
#     print(f"{key} - delta_W mean: {delta_mean:.14f}, std: {delta_std:.4f} | grad mean: {grad_mean:.14f}, std: {grad_std:.4f}")

# for key in delta_W_logs.keys():
#     delta_flat = delta_W_logs[key].flatten()
#     grad_flat = grad_logs[key].flatten()
#     correlation = np.corrcoef(delta_flat, grad_flat)[0, 1]
#     print(f"{key} - Correlation: {correlation:.4f}")

# for key in delta_W_logs.keys():
#     diff = delta_W_logs[key] - grad_logs[key]
#     diff_mean = np.mean(diff)
#     diff_std = np.std(diff)
#     print(f"{key} - Difference mean: {diff_mean:.4f}, std: {diff_std:.4f}")

# for key in delta_W_norms.keys():
#     print(f"{key} - delta_W norm: {delta_W_norms[key]:.4f}, grad norm: {grad_norms[key]:.4f}")