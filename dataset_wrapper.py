"""
自定义数据集包装器，用于解决HuggingFace datasets的索引问题
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class SafeDatasetWrapper(Dataset):
    """
    安全的数据集包装器，将HuggingFace数据集转换为标准的PyTorch数据集
    """
    def __init__(self, hf_dataset, max_samples=None):
        self.hf_dataset = hf_dataset
        self.length = len(hf_dataset)
        
        if max_samples is not None:
            self.length = min(self.length, max_samples)
        
        # 预加载所有数据到内存中（对于CIFAR100这样的小数据集是可行的）
        print(f"[INFO] 预加载 {self.length} 个样本到内存...")
        self.data = []
        
        for i in range(self.length):
            try:
                sample = hf_dataset[i]
                if sample is not None and 'pixel_values' in sample:
                    self.data.append(sample)
                else:
                    print(f"[WARNING] 样本 {i} 无效，跳过")
            except Exception as e:
                print(f"[WARNING] 无法加载样本 {i}: {e}")
                # 继续加载其他样本
                continue
        
        self.actual_length = len(self.data)
        print(f"[INFO] 成功加载 {self.actual_length} 个有效样本")
    
    def __len__(self):
        return self.actual_length
    
    def __getitem__(self, idx):
        if idx >= self.actual_length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.actual_length}")
        
        try:
            return self.data[idx]
        except Exception as e:
            print(f"[ERROR] 访问样本 {idx} 失败: {e}")
            # 返回第一个样本作为fallback
            return self.data[0] if self.data else None

class StableDatasetWrapper(Dataset):
    """
    更轻量的数据集包装器，不预加载所有数据，但提供更安全的访问机制
    """
    def __init__(self, hf_dataset, max_samples=None):
        self.hf_dataset = hf_dataset
        self.length = len(hf_dataset)
        
        if max_samples is not None:
            self.length = min(self.length, max_samples)
        
        print(f"[INFO] 包装数据集，大小: {self.length}")
        
        # 验证前几个样本
        valid_count = 0
        for i in range(min(10, self.length)):
            try:
                sample = hf_dataset[i]
                if sample is not None and 'pixel_values' in sample:
                    valid_count += 1
            except:
                pass
        
        print(f"[INFO] 前10个样本中有 {valid_count} 个有效")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 确保索引在有效范围内
        if idx >= self.length:
            idx = idx % self.length
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sample = self.hf_dataset[idx]
                if sample is not None and 'pixel_values' in sample:
                    return sample
                else:
                    # 如果样本无效，尝试下一个
                    idx = (idx + 1) % self.length
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[ERROR] 多次尝试后仍无法访问样本 {idx}: {e}")
                    # 返回一个默认样本
                    return self._get_default_sample()
                idx = (idx + 1) % self.length
        
        return self._get_default_sample()
    
    def _get_default_sample(self):
        """返回一个默认样本"""
        return {
            'pixel_values': torch.zeros(3, 224, 224),
            'fine_label': 0,
            'label': 0
        }