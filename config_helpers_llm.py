"""
config_helpers_llm.py
=====================
在原 config_helpers.py / config_helpers2.py 的基础上扩展，新增对
LLaMA / Mistral / Qwen2 等 LLM 的剪枝配置支持。

关键设计决策：
  1. GQA (Grouped Query Attention) 剪枝粒度
     - LLaMA2-7B/13B、Mistral-7B 使用 GQA：Q 有 32 heads，KV 只有 8 heads
     - 不能按单个 head 剪，必须以 KV group 为粒度（每组 4 heads，head_dim=128，共 512 维）
     - Q 和 K/V 的剪枝粒度不同：Q granularity = [4*128=512, 4096]，
       K/V granularity = [1*128=128, 4096]（或直接整组剪）
     - 本实现采用"以 KV group 为单位整组剪"策略，Q/K/V 共享 dependency_group_id

  2. SwiGLU FFN 耦合剪枝
     - LLM FFN 结构：out = down_proj(silu(gate_proj(x)) * up_proj(x))
     - gate_proj 和 up_proj 输出维度必须完全一致，不能独立剪枝
     - 实现方式：gate_proj 和 up_proj 共享同一 dependency_group_id，
       down_proj 的输入维度反向跟随

  3. 模块类型检测
     - 原代码用 isinstance(module, BertLayer) 检测，LLM 使用 LlamaDecoderLayer 等
     - 采用懒导入（try/except）避免未安装模型时 ImportError
"""

from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.swin.modeling_swin import SwinLayer

# ------------------------------------------------------------------ #
# 懒导入 LLM 层类型（未安装时降级为 None，不影响原有 BERT/ViT 逻辑）
# ------------------------------------------------------------------ #
try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except ImportError:
    LlamaDecoderLayer = None

try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
except ImportError:
    MistralDecoderLayer = None

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:
    Qwen2DecoderLayer = None

# ------------------------------------------------------------------ #
# 辅助：判断是否是 LLM 解码层
# ------------------------------------------------------------------ #
def _is_llm_decoder_layer(module):
    """返回 True 当且仅当 module 是任一支持的 LLM 解码层。"""
    llm_types = tuple(
        t for t in [LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer]
        if t is not None
    )
    return bool(llm_types) and isinstance(module, llm_types)


def _get_llm_attn_proj_names(model_class_str: str):
    """
    返回 LLM 注意力投影层的名称后缀。
    LLaMA/Mistral/Qwen2 统一使用 q_proj / k_proj / v_proj / o_proj。
    """
    return '.self_attn.q_proj', '.self_attn.k_proj', '.self_attn.v_proj', '.self_attn.o_proj'


def _get_llm_ffn_proj_names(model_class_str: str):
    """
    返回 LLM FFN 投影层的名称后缀。
    LLaMA/Mistral/Qwen2 统一使用 gate_proj / up_proj / down_proj (SwiGLU)。
    """
    return '.mlp.gate_proj', '.mlp.up_proj', '.mlp.down_proj'


def _get_layer_index(name: str) -> int:
    """从 module name 中解析层编号，例如 'model.layers.12' → 12。"""
    parts = name.split('.')
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return int(parts[i])
    return 0


# ====================================================================== #
# get_prune_config_for_attn  （新增 LLM 分支）
# ====================================================================== #
def get_prune_config_for_attn(args, model, prune_params_dict):
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    gqa_kv_groups = prune_params_dict.get('gqa_kv_groups', None)  # LLM GQA 专用
    config_list = []

    model_class_str = str(model.__class__)

    # ------------------------------------------------------------------ #
    # 原有：BERT / ViT / Mask2Former
    # ------------------------------------------------------------------ #
    if 'bert' in model_class_str:
        attention_qkv_str = '.attention.self*'
        attention_output_str = '.attention.output.dense'
        dep_id = -1
        use_legacy = True
    elif 'vit' in model_class_str:
        attention_qkv_str = '.attention.attention*'
        attention_output_str = '.attention.output.dense'
        dep_id = -1
        use_legacy = True
    elif 'mask2former' in model_class_str:
        attention_qkv_str = '.attention.self*'
        attention_output_str = '.attention.output.dense'
        dep_id = -3
        use_legacy = True
    else:
        use_legacy = False  # LLM 路径

    if use_legacy:
        # ---- 原有逻辑（保持不变）----
        for name, module in model.named_modules():
            if 'encoder' in name:
                inc = 0
            else:
                inc = 100

            if isinstance(module, SwinLayer):
                if 'm2f' in args.arch:
                    if '.0.' in name:
                        granularity_ = [32, 128]
                    elif '.1.' in name:
                        granularity_ = [32, 256]
                    elif '.2.' in name:
                        granularity_ = [32, 512]
                    else:
                        granularity_ = [32, 1024]
                else:
                    if '.0.' in name:
                        granularity_ = [32, 192]
                    elif '.1.' in name:
                        granularity_ = [32, 384]
                    elif '.2.' in name:
                        granularity_ = [32, 768]
                    else:
                        granularity_ = [32, 1536]
            else:
                granularity_ = granularity

            if isinstance(module, BertLayer) or isinstance(module, ViTLayer) or isinstance(module, SwinLayer):
                config_list.append({'op_types': ['Linear'],
                                    'op_names_re': [f'{name}{attention_qkv_str}'],
                                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                                    'sparse_ratio': sparse_ratio,
                                    'max_sparse_ratio': max_sparse_ratio,
                                    'granularity': granularity_,
                                    'global_group_id': inc
                                    })
                config_list.append({'op_names': [f'{name}{attention_output_str}'],
                                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                                    'sparse_ratio': sparse_ratio,
                                    'max_sparse_ratio': max_sparse_ratio,
                                    'granularity': list(reversed(granularity_)),
                                    'global_group_id': inc
                                    })
        return config_list

    # ------------------------------------------------------------------ #
    # 新增：LLM 路径（LLaMA / Mistral / Qwen2）
    # ------------------------------------------------------------------ #
    q_str, k_str, v_str, o_str = _get_llm_attn_proj_names(model_class_str)

    for name, module in model.named_modules():
        if not _is_llm_decoder_layer(module):
            continue

        layer_idx = _get_layer_index(name)
        inc = layer_idx  # 每层有唯一 inc，避免 dependency_group_id 冲突
        dep_group_id = layer_idx  # attn 层内 Q/K/V/O 共享同一组

        # ---- 设置 Q 粒度（GQA 情况下以 KV group 为单位）----
        if gqa_kv_groups is not None:
            # Q 剪粒度 = 每个 KV group 对应的 Q head 数 * head_dim
            # 例如 llama2-7b: 32Q/8KV → 4 heads/group * 128 = 512
            q_granularity = granularity  # 已在 config.py 中预算好
            kv_granularity = [granularity[0] // (granularity[0] // granularity[0]),
                               granularity[1]]
            # 实际 K/V granularity: 以整个 KV head 为最小粒度，但绑定到同一 dep group
            kv_granularity = granularity  # 简化：KV 与 Q 使用相同粒度（整组剪）
        else:
            q_granularity = granularity
            kv_granularity = granularity

        # ---- Q proj ----
        config_list.append({
            'op_names': [f'{name}{q_str}'],
            'dependency_group_id': dep_group_id,
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': q_granularity,
            'global_group_id': 0,  # attn 全局组 0
        })
        # ---- K proj ----
        config_list.append({
            'op_names': [f'{name}{k_str}'],
            'dependency_group_id': dep_group_id,
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': kv_granularity,
            'global_group_id': 0,
        })
        # ---- V proj ----
        config_list.append({
            'op_names': [f'{name}{v_str}'],
            'dependency_group_id': dep_group_id,
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': kv_granularity,
            'global_group_id': 0,
        })
        # ---- O proj（输入维度跟随 Q/K/V 输出，granularity 反转）----
        config_list.append({
            'op_names': [f'{name}{o_str}'],
            'dependency_group_id': dep_group_id,
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': list(reversed(kv_granularity)),
            'global_group_id': 0,
        })

    return config_list


# ====================================================================== #
# get_prune_config_for_ffn  （新增 LLM 分支）
# ====================================================================== #
def get_prune_config_for_ffn(args, model, prune_params_dict):
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    coupled_proj = prune_params_dict.get('coupled_proj', False)  # SwiGLU 耦合标志
    config_list = []

    model_class_str = str(model.__class__)

    # ------------------------------------------------------------------ #
    # 原有：BERT / ViT / Mask2Former
    # ------------------------------------------------------------------ #
    if 'bert' in model_class_str:
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id = -1
        use_legacy = True
    elif 'vit' in model_class_str:
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id = -1
        use_legacy = True
    elif 'mask2former' in model_class_str:
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id = -3
        use_legacy = True
    else:
        use_legacy = False

    if use_legacy:
        for name, module in model.named_modules():
            if 'encoder' in name:
                inc = 200
            else:
                inc = 300

            if isinstance(module, SwinLayer):
                if 'm2f' in args.arch:
                    if '.0.' in name:
                        granularity_ = [32, 128]
                    elif '.1.' in name:
                        granularity_ = [32, 256]
                    elif '.2.' in name:
                        granularity_ = [32, 512]
                    else:
                        granularity_ = [32, 1024]
                else:
                    if '.0.' in name:
                        granularity_ = [1, 192]
                    elif '.1.' in name:
                        granularity_ = [1, 384]
                    elif '.2.' in name:
                        granularity_ = [1, 768]
                    else:
                        granularity_ = [1, 1536]
            else:
                granularity_ = granularity

            if isinstance(module, BertLayer) or isinstance(module, ViTLayer) or isinstance(module, SwinLayer):
                config_list.append({'op_names': [f'{name}{intermediate_str}'],
                                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                                    'sparse_ratio': sparse_ratio,
                                    'max_sparse_ratio': max_sparse_ratio,
                                    'granularity': granularity_,
                                    'global_group_id': inc
                                    })
                config_list.append({'op_names': [f'{name}{output_str}'],
                                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                                    'sparse_ratio': sparse_ratio,
                                    'max_sparse_ratio': max_sparse_ratio,
                                    'granularity': list(reversed(granularity_)),
                                    'global_group_id': inc
                                    })
        return config_list

    # ------------------------------------------------------------------ #
    # 新增：LLM 路径（SwiGLU FFN）
    #
    # FFN 结构：out = down_proj( silu(gate_proj(x)) * up_proj(x) )
    # - gate_proj 和 up_proj 输出通道数必须一致（SwiGLU 约束）
    # - 因此 gate_proj / up_proj 共享同一 dependency_group_id（耦合剪枝）
    # - down_proj 的输入通道数必须等于 gate/up 的输出通道数 → 用反转 granularity
    # ------------------------------------------------------------------ #
    gate_str, up_str, down_str = _get_llm_ffn_proj_names(model_class_str)

    # FFN 全局组 ID 偏移（与 attn 的 0 区分开，用 1000 起始）
    FFN_GLOBAL_GROUP_OFFSET = 1000

    for name, module in model.named_modules():
        if not _is_llm_decoder_layer(module):
            continue

        layer_idx = _get_layer_index(name)
        # gate 和 up 共享同一 dep group，down 用另一个 dep group（输入维度依赖）
        ffn_dep_group_id = FFN_GLOBAL_GROUP_OFFSET + layer_idx

        # ---- gate_proj ----
        config_list.append({
            'op_names': [f'{name}{gate_str}'],
            'dependency_group_id': ffn_dep_group_id,  # 与 up_proj 绑定
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': granularity,
            'global_group_id': FFN_GLOBAL_GROUP_OFFSET,
        })
        # ---- up_proj（必须与 gate_proj 耦合，同一 dep group）----
        config_list.append({
            'op_names': [f'{name}{up_str}'],
            'dependency_group_id': ffn_dep_group_id,  # 与 gate_proj 绑定！
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': granularity,
            'global_group_id': FFN_GLOBAL_GROUP_OFFSET,
        })
        # ---- down_proj（输入维度跟随 gate/up 输出 → granularity 反转）----
        config_list.append({
            'op_names': [f'{name}{down_str}'],
            'dependency_group_id': ffn_dep_group_id,
            'sparse_ratio': sparse_ratio,
            'max_sparse_ratio': max_sparse_ratio,
            'granularity': list(reversed(granularity)),
            'global_group_id': FFN_GLOBAL_GROUP_OFFSET,
        })

    return config_list


# ====================================================================== #
# get_prune_config_for_qkv  （原有函数，新增 LLM 分支）
# ====================================================================== #
def get_prune_config_for_qkv(args, model, prune_params_dict):
    """
    为注意力头中的 q、k、v 三个线性层创建剪枝配置（精细化版本）。
    当需要对 Q/K/V 分别指定不同粒度时使用此函数。
    """
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    gqa_kv_groups = prune_params_dict.get('gqa_kv_groups', None)
    config_list = []

    model_class_str = str(model.__class__)

    # ------------------------------------------------------------------ #
    # 原有：BERT / ViT / Mask2Former
    # ------------------------------------------------------------------ #
    if 'bert' in model_class_str:
        q_str, k_str, v_str = '.attention.self.query', '.attention.self.key', '.attention.self.value'
        dep_id = -1
        use_legacy = True
    elif 'vit' in model_class_str:
        q_str, k_str, v_str = '.attention.attention.query', '.attention.attention.key', '.attention.attention.value'
        dep_id = -1
        use_legacy = True
    elif 'mask2former' in model_class_str:
        q_str, k_str, v_str = '.attention.self.query', '.attention.self.key', '.attention.self.value'
        dep_id = -3
        use_legacy = True
    else:
        use_legacy = False

    if use_legacy:
        for name, module in model.named_modules():
            if 'encoder' in name:
                inc = 0
            else:
                inc = 100

            if isinstance(module, SwinLayer):
                if 'm2f' in args.arch:
                    if '.0.' in name:
                        granularity_ = [32, 128]
                    elif '.1.' in name:
                        granularity_ = [32, 256]
                    elif '.2.' in name:
                        granularity_ = [32, 512]
                    else:
                        granularity_ = [32, 1024]
                else:
                    if '.0.' in name:
                        granularity_ = [32, 192]
                    elif '.1.' in name:
                        granularity_ = [32, 384]
                    elif '.2.' in name:
                        granularity_ = [32, 768]
                    else:
                        granularity_ = [32, 1536]
            else:
                granularity_ = granularity

            if isinstance(module, BertLayer) or isinstance(module, ViTLayer) or isinstance(module, SwinLayer):
                config_list.append({
                    'op_types': ['Linear'],
                    'op_names': [f'{name}{q_str}'],
                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                    'sparse_ratio': sparse_ratio,
                    'max_sparse_ratio': max_sparse_ratio,
                    'granularity': granularity_,
                    'global_group_id': inc
                })
                config_list.append({
                    'op_types': ['Linear'],
                    'op_names': [f'{name}{k_str}'],
                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                    'sparse_ratio': sparse_ratio,
                    'max_sparse_ratio': max_sparse_ratio,
                    'granularity': granularity_,
                    'global_group_id': inc
                })
                config_list.append({
                    'op_types': ['Linear'],
                    'op_names': [f'{name}{v_str}'],
                    'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                    'sparse_ratio': sparse_ratio,
                    'max_sparse_ratio': max_sparse_ratio,
                    'granularity': granularity_,
                    'global_group_id': inc
                })
        return config_list

    # ------------------------------------------------------------------ #
    # 新增：LLM 路径
    # 使用 get_prune_config_for_attn 的 LLM 分支即可，此处复用逻辑
    # ------------------------------------------------------------------ #
    q_str, k_str, v_str, _ = _get_llm_attn_proj_names(model_class_str)

    for name, module in model.named_modules():
        if not _is_llm_decoder_layer(module):
            continue

        layer_idx = _get_layer_index(name)
        dep_group_id = layer_idx

        # GQA: Q 粒度 = 整个 KV group，K/V 粒度以单个 KV head 为单位
        if gqa_kv_groups is not None:
            q_gran = granularity
            # K/V head_dim = granularity[0] / (num_q_heads / gqa_kv_groups)
            # 简化：K/V 也使用相同粒度（整组剪枝策略）
            kv_gran = granularity
        else:
            q_gran = kv_gran = granularity

        for proj_str, gran in [(q_str, q_gran), (k_str, kv_gran), (v_str, kv_gran)]:
            config_list.append({
                'op_types': ['Linear'],
                'op_names': [f'{name}{proj_str}'],
                'dependency_group_id': dep_group_id,
                'sparse_ratio': sparse_ratio,
                'max_sparse_ratio': max_sparse_ratio,
                'granularity': gran,
                'global_group_id': 0,
            })

    return config_list
