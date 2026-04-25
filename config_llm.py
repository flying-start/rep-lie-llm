"""
config_llm.py
=============
在原 config.py 的基础上扩展：
  1. hidden_dim 新增 LLM 模型（llama2-7b / llama2-13b / llama3-8b / mistral-7b / qwen2-7b）
  2. training_params 新增 LLM 分支（batch_size=1, gradient_accumulation=16, lr=2e-5）
  3. pruning_params 新增 LLM 分支（GQA/SwiGLU 感知的 granularity）
  4. 新增辅助方法 is_llm_arch()
"""


class Config:
    def __init__(self, args):
        self.core_res = args.core_res
        self.init_sparse_ratio = args.init_sparse_ratio
        self.iter_sparse_ratio = args.iter_sparse_ratio
        self.num_pruning_iters = args.num_pruning_iters

        # ------------------------------------------------------------------ #
        # hidden_dim 映射（新增 LLM）
        # ------------------------------------------------------------------ #
        LLM_HIDDEN_DIM = {
            'llama2-7b':   4096,
            'llama2-13b':  5120,
            'llama3-8b':   4096,
            'mistral-7b':  4096,
            'qwen2-7b':    3584,
        }

        if any(k in args.arch for k in ['bert-base-uncased', 'vit-base', 'm2f']):
            self.hidden_dim = 768
        elif any(k in args.arch for k in ['bert-large-uncased', 'vit-large']):
            self.hidden_dim = 1024
        elif 'vit-huge' in args.arch:
            self.hidden_dim = 1280
        elif args.arch in LLM_HIDDEN_DIM:
            self.hidden_dim = LLM_HIDDEN_DIM[args.arch]
        else:
            # 兜底：尝试从 args.model_path 加载时由外部传入，此处设默认值
            self.hidden_dim = 4096

        global_flag = True

        # ------------------------------------------------------------------ #
        # training_params
        # ------------------------------------------------------------------ #
        self.training_params = {
            # ---------- 默认（BERT 风格）----------
            'model_default': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-5},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-5}
                }
            },
            # ---------- ViT ----------
            'vit-base': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-4}
                }
            },
            'vit-large': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-4}
                }
            },
            'vit-huge': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 5e-5},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 5e-5}
                }
            },
            # ---------- Mask2Former ----------
            'm2f': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-4}
                },
                'cityscapes': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-4}
                },
                'kitti': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 1e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 1e-4}
                },
            },
            # ---------- BERT ----------
            'bert-base-uncased': {
                'data_default': {
                    'init': {'num_train_epochs': 3, 'learning_rate': 5e-4},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 2e-5}
                }
            },
            'bert-large-uncased': {
                'data_default': {
                    'init': {'num_train_epochs': 2, 'learning_rate': 2e-5},
                    'iter': {'num_train_epochs': 2, 'learning_rate': 2e-5}
                }
            },
            # ================================================================ #
            # LLM 分支（新增）
            # 说明：
            #   - batch_size=1  + gradient_accumulation=16 → 等效 batch 16
            #   - 校准阶段（init）不做微调，仅用少量步骤收集重要性分数
            #   - 迭代阶段（iter）用 LoRA 进行恢复微调
            # ================================================================ #
            'llama2-7b': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    },
                    'iter': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    }
                }
            },
            'llama2-13b': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 1e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 32,
                    },
                    'iter': {
                        'num_train_epochs': 1,
                        'learning_rate': 1e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 32,
                    }
                }
            },
            'llama3-8b': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    },
                    'iter': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    }
                }
            },
            'mistral-7b': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    },
                    'iter': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    }
                }
            },
            'qwen2-7b': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    },
                    'iter': {
                        'num_train_epochs': 1,
                        'learning_rate': 2e-5,
                        'batch_size': 1,
                        'gradient_accumulation': 16,
                    }
                }
            },
        }

        # ------------------------------------------------------------------ #
        # pruning_params
        # ------------------------------------------------------------------ #
        self.pruning_params = {
            # ---- 默认 (BERT / ViT) ----
            'model_default': {
                'data_default': {
                    'init': {
                        'training_steps': 10,
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.85,
                                 'granularity': [self.core_res, self.hidden_dim]},
                        'ffn':  {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.85,
                                 'granularity': [1, self.hidden_dim]}
                    },
                    'iter': {
                        'training_steps': 10,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, self.hidden_dim]},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, self.hidden_dim]}
                    }
                }
            },
            # ================================================================ #
            # LLM 剪枝参数（新增）
            #
            # GQA 注意力：LLaMA2-7B 有 32 个 Q head，8 个 KV head
            #   → granularity 的 head 维度应对齐 KV group 数（32//8=4 heads/group）
            #   → attn granularity = [num_kv_groups * head_dim, hidden_dim]
            #     = [4 * 128, 4096] = [512, 4096]
            #
            # SwiGLU FFN：gate_proj 和 up_proj 必须耦合剪枝
            #   → 在 config_helpers_llm.py 中通过 dependency_group_id 绑定
            #   → granularity = [1, hidden_dim] 即按行（neuron）剪枝
            # ================================================================ #
            'llama2-7b': {
                'data_default': {
                    'init': {
                        'training_steps': 128,          # 校准步数（对应 256 条样本 / batch=2）
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {
                            'sparse_ratio': self.init_sparse_ratio,
                            'max_sparse_ratio': 0.80,
                            # GQA: 32Q/8KV → 每组 4 heads * head_dim 128 = 512
                            # granularity[0] 必须是 KV group size 的整数倍
                            'granularity': [512, 4096],
                            'gqa_kv_groups': 8,         # 供 config_helpers_llm.py 使用
                        },
                        'ffn': {
                            'sparse_ratio': self.init_sparse_ratio,
                            'max_sparse_ratio': 0.80,
                            'granularity': [1, 4096],   # SwiGLU FFN 按 neuron 剪
                            'coupled_proj': True,        # gate_proj/up_proj 耦合标志
                        }
                    },
                    'iter': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [512, 4096],
                                 'gqa_kv_groups': 8},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, 4096],
                                 'coupled_proj': True}
                    }
                }
            },
            # mistral-7b 与 llama2-7b 相同 GQA (32Q/8KV)，FFN 也是 SwiGLU
            'mistral-7b': {
                'data_default': {
                    'init': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [512, 4096],
                                 'gqa_kv_groups': 8},
                        'ffn':  {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [1, 4096],
                                 'coupled_proj': True}
                    },
                    'iter': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [512, 4096],
                                 'gqa_kv_groups': 8},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, 4096],
                                 'coupled_proj': True}
                    }
                }
            },
            # llama3-8b (32Q/8KV, hidden=4096)
            'llama3-8b': {
                'data_default': {
                    'init': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [512, 4096],
                                 'gqa_kv_groups': 8},
                        'ffn':  {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [1, 4096],
                                 'coupled_proj': True}
                    },
                    'iter': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [512, 4096],
                                 'gqa_kv_groups': 8},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, 4096],
                                 'coupled_proj': True}
                    }
                }
            },
            # llama2-13b (40Q/40KV MHA → 直接用 head_dim=128 做粒度，hidden=5120)
            'llama2-13b': {
                'data_default': {
                    'init': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [128, 5120]},
                        'ffn':  {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [1, 5120],
                                 'coupled_proj': True}
                    },
                    'iter': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [128, 5120]},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, 5120],
                                 'coupled_proj': True}
                    }
                }
            },
            # qwen2-7b (hidden=3584, GQA: 28Q/4KV, head_dim=128, group_size=7)
            'qwen2-7b': {
                'data_default': {
                    'init': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': self.num_pruning_iters,
                        'attn': {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [896, 3584],   # 7 heads * 128
                                 'gqa_kv_groups': 4},
                        'ffn':  {'sparse_ratio': self.init_sparse_ratio,
                                 'max_sparse_ratio': 0.80,
                                 'granularity': [1, 3584],
                                 'coupled_proj': True}
                    },
                    'iter': {
                        'training_steps': 128,
                        'global_flag': global_flag,
                        'num_iters': 1,
                        'attn': {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [896, 3584],
                                 'gqa_kv_groups': 4},
                        'ffn':  {'sparse_ratio': self.iter_sparse_ratio,
                                 'granularity': [1, 3584],
                                 'coupled_proj': True}
                    }
                }
            },
        }

    # ------------------------------------------------------------------ #
    # 辅助属性
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_llm_arch(arch: str) -> bool:
        """判断当前架构是否属于 LLM（causal LM）类型。"""
        llm_prefixes = ('llama2', 'llama3', 'mistral', 'qwen2')
        return any(arch.startswith(p) for p in llm_prefixes)

    # ------------------------------------------------------------------ #
    # Getter 方法（与原始保持接口一致）
    # ------------------------------------------------------------------ #
    def get_init_training_params(self, model_name, data_name):
        default = self.training_params.get(model_name,
                  self.training_params['model_default'])['data_default']['init']
        specific = self.training_params.get(model_name,
                   self.training_params['model_default']).get(data_name, {'init': {}})['init']
        return default | specific

    def get_iter_training_params(self, model_name, data_name):
        default = self.training_params.get(model_name,
                  self.training_params['model_default'])['data_default']['iter']
        specific = self.training_params.get(model_name,
                   self.training_params['model_default']).get(data_name, {'iter': {}})['iter']
        return default | specific

    def get_init_pruning_params(self, model_name, data_name):
        default = self.pruning_params.get(model_name,
                  self.pruning_params['model_default'])['data_default']['init']
        specific = self.pruning_params.get(model_name,
                   self.pruning_params['model_default']).get(data_name, {'init': {}})['init']
        return default | specific

    def get_iter_pruning_params(self, model_name, data_name):
        default = self.pruning_params.get(model_name,
                  self.pruning_params['model_default'])['data_default']['iter']
        specific = self.pruning_params.get(model_name,
                   self.pruning_params['model_default']).get(data_name, {'iter': {}})['iter']
        return default | specific
