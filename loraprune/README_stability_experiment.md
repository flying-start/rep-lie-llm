# 稳定性剪枝实验指南

本文档提供了使用稳定性剪枝训练器进行实验的相关说明。稳定性剪枝是一种通过分析注意力头和FFN层在多次计算中的重要性排名稳定性，来指导剪枝决策的方法。

## 基本原理

1. **稳定性分数**: 在剪枝周期的后期阶段，收集多次计算的重要性分数，分析每个注意力头或FFN神经元在不同批次数据上的排名稳定性。
2. **关键假设**: 重要性排名稳定的参数对模型更为重要，应优先保留。
3. **实现方式**: 通过计算排名偏差来量化稳定性，并将其与原始重要性分数融合，用于剪枝决策。

## 准备工作

确保已安装所有必要的依赖项：

```bash
pip install -r requirements.txt
```

## 运行实验

### 基本使用

稳定性剪枝实验通过`main9.py`文件运行，基本命令如下：

```bash
python loraprune/main9.py \
  --model_name_or_path bert-base-uncased \
  --task_name sst2 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./results/stability_experiment \
  --use_stability True \
  --stability_components attention ffn \
  --stability_weight 0.3 \
  --prune_metric lora \
  --lora_r 8 \
  --mac 0.5
```

### 稳定性相关参数

以下参数可用于调整稳定性剪枝行为：

- `--use_stability`: 是否使用稳定性分数 (默认: True)
- `--stability_components`: 应用稳定性的组件列表，可以是 `attention`, `ffn` 或两者 (默认: `attention ffn`)
- `--stability_weight`: 稳定性分数的权重，范围 0-1 (默认: 0.3)
- `--stability_collection_ratio`: 用于收集稳定性数据的剪枝周期比例 (默认: 0.25，即周期的后1/4)

### 消融实验设置

为了全面评估稳定性分数的有效性，建议进行以下消融实验：

1. **基线实验**：不使用稳定性分数
```bash
python loraprune/main9.py --task_name sst2 --use_stability False --output_dir ./results/baseline
```

2. **仅注意力头稳定性**：稳定性仅应用于注意力机制
```bash
python loraprune/main9.py --task_name sst2 --use_stability True --stability_components attention --output_dir ./results/attn_only
```

3. **仅FFN稳定性**：稳定性仅应用于前馈网络层
```bash
python loraprune/main9.py --task_name sst2 --use_stability True --stability_components ffn --output_dir ./results/ffn_only
```

4. **完整稳定性**：稳定性应用于注意力和FFN层
```bash
python loraprune/main9.py --task_name sst2 --use_stability True --stability_components attention ffn --output_dir ./results/full_stability
```

5. **稳定性权重变化**：尝试不同的稳定性权重值
```bash
python loraprune/main9.py --task_name sst2 --use_stability True --stability_weight 0.1 --output_dir ./results/weight_0.1
python loraprune/main9.py --task_name sst2 --use_stability True --stability_weight 0.5 --output_dir ./results/weight_0.5
python loraprune/main9.py --task_name sst2 --use_stability True --stability_weight 0.7 --output_dir ./results/weight_0.7
```

## 分析结果

运行实验后，可以查看以下文件和目录进行分析：

1. **稳定性日志**: `head_stability_log.csv` - 记录每次应用稳定性时的模块稳定性
2. **稳定性分布**: `stability_distribution.npz` - 包含稳定性分数的数值分布
3. **稳定性报告**: `detailed_stability_report.md` - 详细分析稳定性对不同模块的影响
4. **性能报告**: `final_training_report.txt` - 训练结束后的总体性能报告

## 进阶分析

训练结束后，可以使用以下代码加载稳定性数据进行更深入的分析：

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载稳定性数据
data = np.load("results/full_stability/stability_distribution.npz", allow_pickle=True)
stability_data = data['data'].item()

# 分析注意力头稳定性
attn_modules = stability_data['attention']
for module_name, module_data in attn_modules.items():
    stability_values = module_data['stability_values']
    if stability_values:
        last_stability = stability_values[-1]  # 获取最后一次的稳定性分数
        
        # 绘制稳定性分布图
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(last_stability)), sorted(last_stability))
        plt.title(f"稳定性分布 - {module_name}")
        plt.xlabel("头索引 (按稳定性排序)")
        plt.ylabel("稳定性分数")
        plt.savefig(f"stability_dist_{module_name.replace('.', '_')}.png")
```

## 常见问题

1. **为什么我的稳定性分数都是0？**
   - 确保`stability_collection_ratio`不是太小，至少需要收集3个样本才能计算稳定性。
   - 验证`prune_freq`设置合理，太大的值会导致收集样本不足。

2. **稳定性计算耗时太长怎么办？**
   - 可以尝试减小`stability_collection_ratio`值，或者仅对注意力头应用稳定性。

3. **如何判断稳定性剪枝是否有效？**
   - 对比不同实验设置下的最终模型性能（准确率、F1等）。
   - 比较相同稀疏度下的模型性能变化。
   - 分析详细稳定性报告中的头/神经元重要性排名。 