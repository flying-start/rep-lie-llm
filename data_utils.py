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


def prepare_datasets(model_name: str, task_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    if task_name == 'glue':
        return prepare_datasets_glue(model_name, data_name, tokenizer, cache_dir, eval_key)
    elif task_name == 'img_class':
        if 'cifar' in data_name:
            return prepare_datasets_cifar(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif data_name == 'tinyimagenet' or data_name == 'tiny-imagenet':
            return prepare_datasets_tinyimagenet(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError
    elif task_name == 'img_seg':
        if 'cityscapes' in data_name:
            return prepare_datasets_cityscapes(model_name, data_name, tokenizer, cache_dir, eval_key)
        elif 'kitti' in data_name:
            return prepare_datasets_kitti(model_name, data_name, tokenizer, cache_dir, eval_key)
        else:
            raise NotImplementedError


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

    # used to preprocess the raw data
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            # 对于STSB任务，标签是浮点数，不需要过滤
            if data_name == 'stsb':
                result['labels'] = examples['label']
            else:
                result['labels'] = [
                    label if label in [0, 1, 2] else -100 for label in examples['label']
                ]
                print("Processed label distribution:", set(result['labels']))
        return result
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    raw_datasets = load_dataset("glue", data_name,cache_dir=cache_dir)
    # raw_datasets = load_dataset(f"glue/{data_name}")

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
            validation_datasets = {
                'test': processed_datasets['test']
            }
        else:
            validation_datasets = {
                'validation': processed_datasets['validation']
            }

    return processed_datasets['train'], validation_datasets, None

def avg_seq_length(task_name):
    # Dev set
    TASK_TO_SEQ_LEN = {
        "stsb": 31.47,
        "mrpc": 53.24,
        "rte": 64.59,
        "sst2": 25.16,
        "qqp": 30.55,
        "qnli": 50.97,
        "cola": 11.67,
        "mnli": 39.05,
    }
    return TASK_TO_SEQ_LEN[task_name]




def prepare_datasets_cifar(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[INFO] 尝试加载{data_name}数据集 (第{attempt+1}次)")
            # 添加下载配置以提高稳定性
            train_ds, test_ds = load_dataset(
                data_name, 
                cache_dir=cache_dir, 
                split=['train', 'test'],
                verification_mode='no_checks'  # 跳过验证以避免损坏的缓存问题
            )
            print(f"[INFO] 原始训练集大小: {len(train_ds)}, 测试集大小: {len(test_ds)}")
            
            # split up training into training + validation with stratification for better balance
            splits = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column='fine_label' if data_name == 'cifar100' else 'label')
            train_ds = splits['train']
            val_ds = splits['test']
            
            print(f"[INFO] 分割后训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")
            break  # 成功加载，退出重试循环
            
        except Exception as e:
            print(f"[ERROR] CIFAR数据集加载失败 (第{attempt+1}次): {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] 等待3秒后重试...")
                time.sleep(3)
                # 清理可能损坏的缓存
                import shutil
                try:
                    cache_path = os.path.join(cache_dir, data_name)
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                        print(f"[INFO] 已清理缓存目录: {cache_path}")
                except:
                    pass
            else:
                raise e

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224

    normalize = Normalize(mean=image_mean, std=image_std)
    #对输入数据标准化：随机裁剪图像为224*224、随机水平翻转图像、转换为tensor、并将像素值从 [0, 255] 缩放到 [0, 1] 的浮点数范围
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transform(examples):
        try:
            examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
            return examples
        except Exception as e:
            print(f"[ERROR] 训练数据变换失败: {e}")
            # 返回空的batch以避免训练中断
            return {'pixel_values': [], 'fine_label': [] if 'fine_label' in examples else [], 'label': [] if 'label' in examples else []}

    def val_transform(examples):
        try:
            examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
            return examples
        except Exception as e:
            print(f"[ERROR] 验证数据变换失败: {e}")
            # 返回空的batch以避免验证中断
            return {'pixel_values': [], 'fine_label': [] if 'fine_label' in examples else [], 'label': [] if 'label' in examples else []}

    try:
        # 使用更简单的transform方式，避免批量处理问题
        print(f"[INFO] 设置数据变换函数...")
        
        # 直接设置transform而不是map
        train_ds.set_transform(train_transform)
        val_ds.set_transform(val_transform)  
        test_ds.set_transform(val_transform)
        
        print(f"[INFO] 数据变换设置完成")
        print(f"[INFO] 最终训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}, 测试集大小: {len(test_ds)}")
        
        # 测试数据集是否正常工作
        print(f"[INFO] 测试数据集访问...")
        try:
            sample = train_ds[0]
            print(f"[INFO] 成功访问训练集样本，shape: {sample['pixel_values'].shape if 'pixel_values' in sample else 'No pixel_values'}")
        except Exception as test_e:
            print(f"[ERROR] 数据集访问测试失败: {test_e}")
            raise test_e
        
    except Exception as e:
        print(f"[ERROR] 数据变换设置失败: {e}")
        raise e

    return train_ds, val_ds, test_ds


def prepare_datasets_tinyimagenet(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[INFO] 尝试加载{data_name}数据集 (第{attempt+1}次)")
            # TinyImageNet有200个类别
            train_ds, test_ds = load_dataset(
                'Maysee/tiny-imagenet', 
                cache_dir=cache_dir, 
                split=['train', 'valid'],
                verification_mode='no_checks'
            )
            print(f"[INFO] 原始训练集大小: {len(train_ds)}, 测试集大小: {len(test_ds)}")
            
            # split up training into training + validation with stratification
            splits = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
            train_ds = splits['train']
            val_ds = splits['test']
            
            print(f"[INFO] 分割后训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")
            break  # 成功加载，退出重试循环
            
        except Exception as e:
            print(f"[ERROR] TinyImageNet数据集加载失败 (第{attempt+1}次): {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] 等待3秒后重试...")
                time.sleep(3)
                # 清理可能损坏的缓存
                import shutil
                try:
                    cache_path = os.path.join(cache_dir, 'Maysee___tiny-imagenet')
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                        print(f"[INFO] 已清理缓存目录: {cache_path}")
                except:
                    pass
            else:
                raise e

    image_mean, image_std = tokenizer.image_mean, tokenizer.image_std
    size = 224

    normalize = Normalize(mean=image_mean, std=image_std)
    # TinyImageNet的增强策略
    _train_transforms = Compose(
        [
            RandomResizedCrop(size, scale=(0.8, 1.0)),  # 更强的裁剪
            RandomHorizontalFlip(p=0.5),
            RandomRotation(15),  # 添加旋转
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize((size, size)),  # 直接resize到目标大小
            ToTensor(),
            normalize,
        ]
    )

    def train_transform(examples):
        try:
            examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
            return examples
        except Exception as e:
            print(f"[ERROR] TinyImageNet训练数据变换失败: {e}")
            # 返回空的batch以避免训练中断
            return {'pixel_values': [], 'label': [] if 'label' in examples else []}

    def val_transform(examples):
        try:
            examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
            return examples
        except Exception as e:
            print(f"[ERROR] TinyImageNet验证数据变换失败: {e}")
            # 返回空的batch以避免验证中断
            return {'pixel_values': [], 'label': [] if 'label' in examples else []}

    try:
        # 设置数据变换函数
        print(f"[INFO] 设置TinyImageNet数据变换函数...")
        
        train_ds.set_transform(train_transform)
        val_ds.set_transform(val_transform)  
        test_ds.set_transform(val_transform)
        
        print(f"[INFO] TinyImageNet数据变换设置完成")
        print(f"[INFO] 最终训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}, 测试集大小: {len(test_ds)}")
        
        # 测试数据集是否正常工作
        print(f"[INFO] 测试TinyImageNet数据集访问...")
        try:
            sample = train_ds[0]
            print(f"[INFO] 成功访问TinyImageNet训练集样本，shape: {sample['pixel_values'].shape if 'pixel_values' in sample else 'No pixel_values'}")
            print(f"[INFO] 标签: {sample.get('label', 'unknown')}")
        except Exception as test_e:
            print(f"[ERROR] TinyImageNet数据集访问测试失败: {test_e}")
            raise test_e
        
    except Exception as e:
        print(f"[ERROR] TinyImageNet数据变换设置失败: {e}")
        raise e

    return train_ds, val_ds, test_ds


def prepare_datasets_cityscapes(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    num_classes = 19

    train_tokenizer = copy(tokenizer)
    train_mul = 1
    train_tokenizer.size = {'height': int(512 * train_mul), 'width': int(1024 * train_mul)}
    eval_tokenizer = copy(tokenizer)
    eval_mul = 2
    eval_tokenizer.size = {'height': int(512 * eval_mul), 'width': int(1024 * eval_mul)}

    train_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/train.lst',
        tokenizer=train_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    val_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/val.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    test_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/cityscapes/test.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    return train_ds, val_ds, test_ds


def prepare_datasets_kitti(model_name: str, data_name: str, tokenizer, cache_dir: str, eval_key: str = 'val'):
    num_classes = 19

    train_tokenizer = copy(tokenizer)
    train_mul = 1
    train_tokenizer.size = {'height': int(375 * train_mul), 'width': int(1242 * train_mul)}
    eval_tokenizer = copy(tokenizer)
    eval_mul = 1
    eval_tokenizer.size = {'height': int(375 * eval_mul), 'width': int(1242 * eval_mul)}

    train_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/kitti/train.lst',
        tokenizer=train_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    val_ds = Cityscapes(
        root=cache_dir,
        list_path='/list/kitti/val.lst',
        tokenizer=eval_tokenizer,
        num_classes=num_classes,
        ignore_label=255)

    return train_ds, val_ds, val_ds


class Cityscapes(data.Dataset):
    def __init__(self,
                 root,
                 list_path,
                 tokenizer,
                 num_classes=19,
                 ignore_label=255):

        super(Cityscapes, self).__init__()

        self.tokenizer = tokenizer
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

        self.target_mode = False
        self.image_mode = False

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
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
        if 'cityscapes' in self.list_path:
            folder = 'cityscapes'
        else:
            folder = 'kitti'

        image = cv2.imread(os.path.join(self.root, folder, item["img"]), cv2.IMREAD_COLOR)

        if 'test' in self.list_path:
            return self.tokenizer(image)

        label = cv2.imread(os.path.join(self.root, folder, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        return self.tokenizer(image, label)
