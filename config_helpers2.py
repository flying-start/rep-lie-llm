from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.swin.modeling_swin import SwinLayer


def get_prune_config_for_attn(args, model, prune_params_dict):
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    config_list = []

    if 'bert' in str(model.__class__):
        attention_qkv_str = '.attention.self*'
        attention_output_str = '.attention.output.dense'
        dep_id = -1
    elif 'vit' in str(model.__class__):
        attention_qkv_str = '.attention.attention*'
        attention_output_str = '.attention.output.dense'
        dep_id = -1
    elif 'mask2former' in str(model.__class__):
        attention_qkv_str = '.attention.self*'
        attention_output_str = '.attention.output.dense'
        dep_id = -3
    else:
        raise NotImplementedError

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


def get_prune_config_for_ffn(args, model, prune_params_dict):
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    config_list = []

    if 'bert' in str(model.__class__):
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id =  -1
    elif 'vit' in str(model.__class__):
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id = -1
    elif 'mask2former' in str(model.__class__):
        intermediate_str = '.intermediate.dense'
        output_str = '.output.dense'
        dep_id = -3
    else:
        raise NotImplementedError

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



def get_prune_config_for_qkv(args, model, prune_params_dict):
    """
    为注意力头中的 q、k、v 三个线性层创建剪枝配置。
    """
    sparse_ratio = prune_params_dict['sparse_ratio']
    max_sparse_ratio = prune_params_dict.get('max_sparse_ratio', 1)
    granularity = prune_params_dict['granularity']
    config_list = []

    # 确定模型类型并设置规则
    if 'bert' in str(model.__class__):
        q_str, k_str, v_str = '.attention.self.query', '.attention.self.key', '.attention.self.value'
        dep_id = -1
    elif 'vit' in str(model.__class__):
        q_str, k_str, v_str = '.attention.attention.query', '.attention.attention.key', '.attention.attention.value'
        dep_id = -1
    elif 'mask2former' in str(model.__class__):
        q_str, k_str, v_str = '.attention.self.query', '.attention.self.key', '.attention.self.value'
        dep_id = -3
    else:
        raise NotImplementedError

    # 遍历模型模块，定位需要剪枝的注意力层
    for name, module in model.named_modules():

        # 增量，用于处理全局组 ID（避免重复）
        if 'encoder' in name:
            inc = 0
        else:
            inc = 100

        # 设置 Swin 模型的特定 granularity
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

        # 针对每层 Transformer 层
        if isinstance(module, BertLayer) or isinstance(module, ViTLayer) or isinstance(module, SwinLayer):
            # 针对 query
            config_list.append({
                'op_types': ['Linear'],
                'op_names': [f'{name}{q_str}'],
                'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                'sparse_ratio': sparse_ratio,
                'max_sparse_ratio': max_sparse_ratio,
                'granularity': granularity_,
                'global_group_id': inc
            })
            # 针对 key
            config_list.append({
                'op_types': ['Linear'],
                'op_names': [f'{name}{k_str}'],
                'dependency_group_id': int(name.split('.')[dep_id]) + inc,
                'sparse_ratio': sparse_ratio,
                'max_sparse_ratio': max_sparse_ratio,
                'granularity': granularity_,
                'global_group_id': inc
            })
            # 针对 value
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
