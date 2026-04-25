"""
data_utils_llm.py
=================
在原 data_utils.py 的基础上新增对 LLM 校准/评估数据集的支持：
  - prepare_datasets() 新增 'llm_causal' 任务分支
  - 新增 prepare_datasets_llm_causal() 函数
    * 校准集：C4 流式加载（避免下载整个数据集），截取 calibration_nsamples 条 2048-token 序列
    * 评估集：WikiText-2 test split，用于计算 Perplexity (PPL)

原有 GLUE / CIFAR / TinyImageNet 等函数保持不变。
"""

from __future__ import annotations

import os
from copy import copy

import cv2
from datasets import load_dataset
from torch.utils import data
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    RandomRotation,
                                    ColorJitter,
                                    Resize,
                                    ToTensor)

skip_exec = True


# ====================================================================== #
# 顶层分发函数（新增 llm_causal 分支）
# ====================================================================== #
def prepare_datasets(model_name: str, task_name: str, data_name: str, tokenizer,
                     cache_dir: str, eval_key: str = 'val',
                     # LLM 专属参数（非 LLM 任务时忽略）
                     calibration_nsamples: int = 256,
                     max_seq_length: int = 2048):
    if task_name == 'glue':
        return prepare_datasets_glue(model_name, data_name, tokenizer, cache_dir, eval_key)
    elif task_name == 'img_class':
        if 'cifar' in data_name:
            return prepare_datasets_cifar(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif data_name in ('tinyimagenet', 'tiny-imagenet'):
            return prepare_datasets_tinyimagenet(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError(f"Unknown img_class dataset: {data_name}")
    elif task_name == 'img_seg':
        if 'cityscapes' in data_name:
            return prepare_datasets_cityscapes(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif 'kitti' in data_name:
            return prepare_datasets_kitti(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError(f"Unknown img_seg dataset: {data_name}")
    # ------------------------------------------------------------------ #
    # 新增：LLM 因果语言模型校准/评估数据集
    # ------------------------------------------------------------------ #
    elif task_name == 'llm_causal':
        return prepare_datasets_llm_causal(
            model_name=model_name,
            calibration_dataset=data_name,   # 通常传 'c4'
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            calibration_nsamples=calibration_nsamples,
            max_seq_length=max_seq_length,
        )
    else:
        raise NotImplementedError(f"Unknown task: {task_name}")


# ====================================================================== #
# 新增：LLM 校准 + PPL 评估数据集
# ====================================================================== #
def prepare_datasets_llm_causal(
        model_name: str,
        calibration_dataset: str,
        tokenizer,
        cache_dir: str,
        calibration_nsamples: int = 256,
        max_seq_length: int = 2048,
        eval_dataset: str = 'wikitext2',
):
    """
    为 LLM 剪枝准备校准集和 PPL 评估集。

    Parameters
    ----------
    model_name : str
        模型名称（暂时只用于日志）。
    calibration_dataset : str
        校准集来源，支持 'c4' / 'wikitext2' / 'ptb'。
    tokenizer : PreTrainedTokenizer
        LLM tokenizer（已加载）。
    cache_dir : str
        HuggingFace 数据集缓存目录。
    calibration_nsamples : int
        从校准集中截取的样本数（每条 max_seq_length tokens），默认 256。
    max_seq_length : int
        每条序列的 token 长度，默认 2048。
    eval_dataset : str
        PPL 评估数据集，默认 'wikitext2'。

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
        train_dataset : TokenizedCalibrationDataset  校准集（用于重要性估计）
        val_dataset   : None  （LLM 无单独验证集，PPL 直接在 test 上算）
        test_dataset  : TokenizedCalibrationDataset  WikiText-2 test（PPL 评估）
    """
    import torch
    import random

    print(f"[INFO] 准备 LLM 校准集: {calibration_dataset}, nsamples={calibration_nsamples}, "
          f"seq_len={max_seq_length}")

    # ------------------------------------------------------------------ #
    # 1. 校准集（C4 / WikiText2 / PTB）
    # ------------------------------------------------------------------ #
    calib_tokens = _load_and_tokenize_calib(
        dataset_name=calibration_dataset,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        nsamples=calibration_nsamples,
        seq_len=max_seq_length,
    )
    train_dataset = TokenizedCalibrationDataset(calib_tokens)

    # ------------------------------------------------------------------ #
    # 2. PPL 评估集（WikiText-2 test）
    # ------------------------------------------------------------------ #
    eval_tokens = _load_and_tokenize_eval(
        dataset_name=eval_dataset,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        seq_len=max_seq_length,
    )
    test_dataset = TokenizedCalibrationDataset(eval_tokens)

    print(f"[INFO] 校准集大小: {len(train_dataset)} 条序列")
    print(f"[INFO] 评估集大小: {len(test_dataset)} 条序列")

    return train_dataset, None, test_dataset


def _load_and_tokenize_calib(dataset_name: str, tokenizer, cache_dir: str,
                              nsamples: int, seq_len: int):
    """
    从指定数据集中流式加载文本，分词后拼接并切分为 nsamples 条 seq_len-token 序列。
    使用流式加载（streaming=True）避免下载整个数据集（C4 约 300+ GB）。
    """
    import torch
    import random

    print(f"[INFO] 流式加载校准数据集: {dataset_name} ...")

    if dataset_name == 'c4':
        raw_ds = load_dataset(
            'allenai/c4',
            'en',
            split='train',
            streaming=True,
            cache_dir=cache_dir,
        )
        text_field = 'text'
    elif dataset_name == 'wikitext2':
        raw_ds = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split='train',
            streaming=True,
            cache_dir=cache_dir,
        )
        text_field = 'text'
    elif dataset_name == 'ptb':
        raw_ds = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='train',
            streaming=True,
            cache_dir=cache_dir,
        )
        text_field = 'sentence'
    else:
        raise NotImplementedError(f"Unsupported calibration dataset: {dataset_name}")

    # 逐条流式读取，拼接 token id，直到积累足够多的序列
    all_ids = []
    samples = []

    # 取 nsamples * 10 条文本以保证足够量（短文本场景下需要更多）
    fetch_limit = nsamples * 20

    for i, example in enumerate(raw_ds):
        text = example[text_field].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

        # 每当积累满 seq_len 个 token 就存一条样本
        while len(all_ids) >= seq_len:
            chunk = all_ids[:seq_len]
            samples.append(torch.tensor(chunk, dtype=torch.long))
            all_ids = all_ids[seq_len:]
            if len(samples) >= nsamples:
                break

        if len(samples) >= nsamples:
            break

        if i >= fetch_limit:
            print(f"[WARNING] 已读取 {i} 条文本，但仅得到 {len(samples)} 条校准序列（需要 {nsamples}）")
            break

    if len(samples) < nsamples:
        print(f"[WARNING] 校准集不足 {nsamples} 条，实际: {len(samples)}，将重复填充")
        while len(samples) < nsamples:
            samples.append(samples[len(samples) % len(samples)])

    # 随机打乱
    random.shuffle(samples)
    return samples[:nsamples]


def _load_and_tokenize_eval(dataset_name: str, tokenizer, cache_dir: str, seq_len: int):
    """
    加载完整的 PPL 评估数据集，拼接后切分为 seq_len-token 序列。
    WikiText-2 test split 约 250K tokens，全量加载（非流式）。
    """
    import torch

    print(f"[INFO] 加载 PPL 评估数据集: {dataset_name} (test split) ...")

    if dataset_name == 'wikitext2':
        raw_ds = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            split='test',
            cache_dir=cache_dir,
        )
        text_field = 'text'
    elif dataset_name == 'ptb':
        raw_ds = load_dataset(
            'ptb_text_only',
            'penn_treebank',
            split='test',
            cache_dir=cache_dir,
        )
        text_field = 'sentence'
    elif dataset_name == 'c4':
        raw_ds = load_dataset(
            'allenai/c4',
            'en',
            split='validation',
            cache_dir=cache_dir,
        )
        text_field = 'text'
    else:
        raise NotImplementedError(f"Unsupported eval dataset: {dataset_name}")

    # 拼接所有文本 → 分词 → 切分序列
    full_text = '\n\n'.join(
        [ex[text_field] for ex in raw_ds if ex[text_field].strip()]
    )
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    samples = []
    for i in range(0, len(token_ids) - seq_len + 1, seq_len):
        chunk = token_ids[i: i + seq_len]
        samples.append(torch.tensor(chunk, dtype=torch.long))

    print(f"[INFO] 评估集共 {len(samples)} 条 {seq_len}-token 序列")
    return samples


class TokenizedCalibrationDataset(data.Dataset):
    """
    简单的 Token-ID 序列数据集，供 LLM 校准和 PPL 评估使用。

    __getitem__ 返回字典：
        {'input_ids': LongTensor[seq_len], 'labels': LongTensor[seq_len]}
    labels 与 input_ids 相同（因果语言建模的 next-token prediction）。
    """

    def __init__(self, token_sequences):
        self.sequences = token_sequences  # List[Tensor]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx]
        return {
            'input_ids': ids,
            'labels': ids.clone(),   # CausalLM 的标签就是 input_ids
        }


# ====================================================================== #
# 以下为原有函数，保持不变
# ====================================================================== #

def prepare_datasets_glue(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    task_to_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }
    sentence1_key, sentence2_key = task_to_keys[data_name]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            if data_name == 'stsb':
                result['labels'] = examples['label']
            else:
                result['labels'] = [
                    label if label in [0, 1, 2] else -100 for label in examples['label']
                ]
                print("Processed label distribution:", set(result['labels']))
        return result

    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    raw_datasets = load_dataset("glue", data_name, cache_dir=cache_dir)

    if eval_key == 'val':
        for key in list(raw_datasets.keys()):
            if 'test' in key:
                raw_datasets.pop(key)
    column_names = raw_datasets['train'].column_names
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names)
    print("Available dataset keys:", raw_datasets.keys())
    if data_name == 'mnli':
        print("Available dataset keys:", processed_datasets.keys())
        if eval_key == 'test':
            validation_datasets = {
                'test_matched': processed_datasets['test_matched'],
                'test_mismatched': processed_datasets['test_mismatched']
            }
        else:
            validation_datasets = {
                'validation_matched': processed_datasets['validation_matched'],
                'validation_mismatched': processed_datasets['validation_mismatched']
            }
    else:
        if eval_key == 'test':
            validation_datasets = {'test': processed_datasets['test']}
        else:
            validation_datasets = {'validation': processed_datasets['validation']}

    return processed_datasets['train'], validation_datasets, None


def avg_seq_length(task_name):
    TASK_TO_SEQ_LEN = {
        "stsb": 31.47, "mrpc": 53.24, "rte": 64.59, "sst2": 25.16,
        "qqp": 30.55, "qnli": 50.97, "cola": 11.67, "mnli": 39.05,
    }
    return TASK_TO_SEQ_LEN[task_name]


def prepare_datasets_cifar(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[INFO] 尝试加载{data_name}数据集 (第{attempt+1}次)")
            train_ds, test_ds = load_dataset(
                data_name,
                cache_dir=cache_dir,
                split=['train', 'test'],
                verification_mode='no_checks'
            )
            splits = train_ds.train_test_split(
                test_size=0.1, seed=42,
                stratify_by_column='fine_label' if data_name == 'cifar100' else 'label'
            )
            train_ds = splits['train']
            val_ds = splits['test']
            break
        except Exception as e:
            print(f"[ERROR] CIFAR数据集加载失败 (第{attempt+1}次): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
                import shutil
                try:
                    cache_path = os.path.join(cache_dir, data_name)
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                except Exception:
                    pass
            else:
                raise

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    _val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def train_transform(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transform(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)
    test_ds.set_transform(val_transform)
    return train_ds, val_ds, test_ds


def prepare_datasets_tinyimagenet(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            train_ds, test_ds = load_dataset(
                'Maysee/tiny-imagenet',
                cache_dir=cache_dir,
                split=['train', 'valid'],
                verification_mode='no_checks'
            )
            splits = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
            train_ds = splits['train']
            val_ds = splits['test']
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose([
        RandomResizedCrop(size, scale=(0.8, 1.0)), RandomHorizontalFlip(p=0.5),
        RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(), normalize,
    ])
    _val_transforms = Compose([Resize((size, size)), ToTensor(), normalize])

    def train_transform(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transform(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    train_ds.set_transform(train_transform)
    val_ds.set_transform(val_transform)
    test_ds.set_transform(val_transform)
    return train_ds, val_ds, test_ds


def prepare_datasets_cityscapes(model_name, data_name, tokenizer, cache_dir, eval_key='val'):
    num_classes = 19
    train_tokenizer = copy(tokenizer)
    train_tokenizer.size = {'height': 512, 'width': 1024}
    eval_tokenizer = copy(tokenizer)
    eval_tokenizer.size = {'height': 1024, 'width': 2048}

    train_ds = Cityscapes(root=cache_dir, list_path='/list/cityscapes/train.lst',
                          tokenizer=train_tokenizer, num_classes=num_classes, ignore_label=255)
    val_ds = Cityscapes(root=cache_dir, list_path='/list/cityscapes/val.lst',
                        tokenizer=eval_tokenizer, num_classes=num_classes, ignore_label=255)
    test_ds = Cityscapes(root=cache_dir, list_path='/list/cityscapes/test.lst',
                         tokenizer=eval_tokenizer, num_classes=num_classes, ignore_label=255)
    return train_ds, val_ds, test_ds


def prepare_datasets_kitti(model_name, data_name, tokenizer, cache_dir, eval_key='val'):
    num_classes = 19
    train_tokenizer = copy(tokenizer)
    train_tokenizer.size = {'height': 375, 'width': 1242}
    eval_tokenizer = copy(tokenizer)
    eval_tokenizer.size = {'height': 375, 'width': 1242}

    train_ds = Cityscapes(root=cache_dir, list_path='/list/kitti/train.lst',
                          tokenizer=train_tokenizer, num_classes=num_classes, ignore_label=255)
    val_ds = Cityscapes(root=cache_dir, list_path='/list/kitti/val.lst',
                        tokenizer=eval_tokenizer, num_classes=num_classes, ignore_label=255)
    return train_ds, val_ds, val_ds


class Cityscapes(data.Dataset):
    def __init__(self, root, list_path, tokenizer, num_classes=19, ignore_label=255):
        super().__init__()
        self.tokenizer = tokenizer
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.files = self.read_files()
        self.label_mapping = {
            -1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3,
            13: 4, 14: ignore_label, 15: ignore_label, 16: ignore_label,
            17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_label,
            30: ignore_label, 31: 16, 32: 17, 33: 18
        }
        self.target_mode = False
        self.image_mode = False

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                name = os.path.splitext(os.path.basename(item[0]))[0]
                files.append({"img": item[0], "name": name})
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({"img": image_path, "label": label_path, "name": name, "weight": 1})
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        folder = 'cityscapes' if 'cityscapes' in self.list_path else 'kitti'
        image = cv2.imread(os.path.join(self.root, folder, item["img"]), cv2.IMREAD_COLOR)
        if 'test' in self.list_path:
            return self.tokenizer(image)
        label = cv2.imread(os.path.join(self.root, folder, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        return self.tokenizer(image, label)
