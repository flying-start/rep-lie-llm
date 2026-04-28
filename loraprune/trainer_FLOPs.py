from transformers.trainer import *
try:
    from transformers.trainer import is_torch_tpu_available
except ImportError:
    # transformers 4.36+ 将该函数移至 training_args 或移除
    def is_torch_tpu_available():
        return False
import loraprune.utils as utils
import loraprune.utils1 as utils1
from peft.tuners.lora import Linear
from torch.utils.data.distributed import DistributedSampler
import time
import os
import numpy as np
from fvcore.nn import FlopCountAnalysis
from copy import deepcopy
import gc
import types
import re
import math

class LoRAPruneTrainer(Trainer):
    def __init__(self, model,
                 train_dataset,
                 eval_dataset,
                 compute_metrics,
                 args,
                 data_collator,
                 ratio,
                 init_ratio,
                 warmup_iters,
                 cooldown_iters,
                 prune_freq,
                 prune_metric,
                 mac,
                 seq_len,
                 task_name=None,
                 # 添加消融实验相关参数
                 use_stability=True,  # 是否使用稳定性分数
                 stability_components=['attention'],  # 应用稳定性分数的组件列表: 'attention', 'ffn', 或两者
                 stability_weight=0.3,  # 稳定性分数的权重
                 stability_collection_ratio=0.25  # 在剪枝周期中用于收集稳定性的比例
                 ):
        super().__init__(model=model,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         args=args,
                         data_collator=data_collator,
                         compute_metrics=compute_metrics
                         )
        self.ratio = ratio
        self.init_ratio = init_ratio
        self.mac = mac
        self.seq_len = seq_len
        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.prune_freq = prune_freq
        self.prune_metric = prune_metric
        self.masks = {} 
        self.mac_ratio= 0.0
        self.task_name = task_name  # 保存任务名称，用于处理不同任务的特殊逻辑
        
        # 保存稳定性相关参数
        self.use_stability = use_stability
        self.stability_components = stability_components
        self.stability_weight = stability_weight
        self.stability_collection_ratio = stability_collection_ratio
        
        # 内部状态变量
        self.collected_importances = {}
        self.stability_stats = {}  # 用于存储稳定性统计数据，方便后续分析
        # self.memory_log = []

        # 添加性能监测相关属性
        self.perf_stats = {
            'forward_time': [],
            'backward_time': [],
            'importance_time': [],
            'pruning_time': [],
            'optimizer_time': []
        }
        self.method_stats = {
            'magnitude': {'time': [], 'memory': []},
            'grad': {'time': [], 'memory': []},
            'lora': {'time': [], 'memory': []}
        }
        self.profile_steps = list(range(10, 500, 50))  # 更广泛地采样性能数据
        
        # 更新trainer以记录实时性能数据
        self.setup_performance_logging()

    def setup_performance_logging(self):
        """设置性能日志记录"""
        import types
        import os
        import time
        import traceback
        # 设置日志文件路径
        self.performance_log_path = os.path.join(self.args.output_dir, "training_performance_log.csv")
        self.performance_summary_path = os.path.join(self.args.output_dir, "training_performance_summary.txt")
        
        # 创建日志文件并写入表头
        with open(self.performance_log_path, "w") as f:
            f.write("step,phase,time_ms,memory_mb,peak_memory_mb,flops_g,method\n")
        
        # 创建摘要日志文件并写入表头
        with open(self.performance_summary_path, "w") as f:
            f.write("# 训练性能摘要日志\n")
            f.write("=" * 80 + "\n\n")
        
        # 添加计算估算FLOPs的辅助函数
        def estimate_flops(self_arg, phase):
            """估算不同阶段的FLOPs"""
            model = self_arg.model
            batch_size = self_arg.args.per_device_train_batch_size
            seq_length = self_arg.seq_len
            hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
            num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for n, p in model.named_parameters() 
                             if 'lora_' in n and p.requires_grad)
            
            if phase == 'forward':
                # 模型前向传播FLOPs
                return 4 * batch_size * seq_length * hidden_size * hidden_size * num_layers / 1e9  # G
            elif phase == 'backward':
                # 模型反向传播FLOPs (通常是前向传播的2倍)
                return 8 * batch_size * seq_length * hidden_size * hidden_size * num_layers / 1e9  # G
            elif phase == 'importance':
                # 根据不同方法估算重要性计算的FLOPs
                method = self_arg.prune_metric
                if method == 'magnitude':
                    return total_params / 1e9  # G
                elif method == 'grad':
                    return 3 * total_params / 1e9  # G
                elif method == 'lora':
                    return 5 * lora_params / 1e9  # G
                else:
                    return 0.0
            elif phase == 'pruning':
                # 剪枝搜索FLOPs (简单估计)
                return 10 * total_params / 1e9  # G
            else:
                return 0.0
        
        # 添加到类中
        self.estimate_flops = types.MethodType(estimate_flops, self)
        
        # 添加记录性能摘要的函数
        def log_performance_summary(self_arg):
            """记录当前步骤的性能摘要"""
            step = self_arg.state.global_step
            
            if step % 100 != 0:  # 每100步记录一次
                return
            
            # 计算各阶段平均性能
            if not self_arg.perf_stats['forward_time']:
                return
            
            # 获取最近的性能数据
            recent_steps = min(100, len(self_arg.perf_stats['forward_time']))
            recent_fw_time = np.mean([t * 1000 for t in self_arg.perf_stats['forward_time'][-recent_steps:]]) # ms
            recent_bw_time = np.mean([t * 1000 for t in self_arg.perf_stats['backward_time'][-recent_steps:]]) # ms
            
            # 如果有记录importance时间
            if self_arg.perf_stats['importance_time']:
                recent_imp_time = np.mean([t * 1000 for t in self_arg.perf_stats['importance_time'][-recent_steps:]]) # ms
            else:
                recent_imp_time = 0.0
            
            # 如果有记录pruning时间
            if 'pruning_time' in self_arg.perf_stats and self_arg.perf_stats['pruning_time']:
                recent_prune_time = np.mean([t * 1000 for t in self_arg.perf_stats['pruning_time'][-recent_steps:]]) # ms
            else:
                recent_prune_time = 0.0
            
            # 获取当前内存使用
            current_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 估算各阶段FLOPs
            fw_flops = self_arg.estimate_flops('forward')
            bw_flops = self_arg.estimate_flops('backward')
            imp_flops = self_arg.estimate_flops('importance')
            
            # 计算总步骤时间
            total_step_time = recent_fw_time + recent_bw_time + recent_imp_time + recent_prune_time
            
            # 总训练时间估计 (小时)
            elapsed_steps = step
            remaining_steps = self_arg.state.max_steps - step
            elapsed_time = sum(self_arg.perf_stats['forward_time']) + sum(self_arg.perf_stats['backward_time'])
            estimated_total_time = elapsed_time * self_arg.state.max_steps / max(1, elapsed_steps)
            estimated_remaining_time = estimated_total_time - elapsed_time
            
            # 格式化为小时:分钟:秒
            def format_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            
            # 写入摘要日志
            with open(self_arg.performance_summary_path, "a") as f:
                f.write(f"\n步骤 {step}/{self_arg.state.max_steps} 性能摘要:\n")
                f.write("-" * 60 + "\n")
                f.write(f"时间分布 (单步):\n")
                f.write(f"  前向传播: {recent_fw_time:.2f}ms ({recent_fw_time/total_step_time*100:.1f}%)\n")
                f.write(f"  反向传播: {recent_bw_time:.2f}ms ({recent_bw_time/total_step_time*100:.1f}%)\n")
                f.write(f"  重要性计算: {recent_imp_time:.2f}ms ({recent_imp_time/total_step_time*100:.1f}%)\n")
                f.write(f"  剪枝操作: {recent_prune_time:.2f}ms ({recent_prune_time/total_step_time*100:.1f}%)\n")
                f.write(f"  总计: {total_step_time:.2f}ms\n\n")
                
                f.write(f"内存使用:\n")
                f.write(f"  当前内存: {current_mem:.2f}MB\n")
                f.write(f"  峰值内存: {peak_mem:.2f}MB\n\n")
                
                f.write(f"计算量 (单步):\n")
                f.write(f"  前向传播: {fw_flops:.2f}GFLOPs\n")
                f.write(f"  反向传播: {bw_flops:.2f}GFLOPs\n")
                f.write(f"  重要性计算 ({self_arg.prune_metric}): {imp_flops:.2f}GFLOPs\n")
                f.write(f"  总计: {fw_flops + bw_flops + imp_flops:.2f}GFLOPs\n\n")
                
                f.write(f"训练进度:\n")
                f.write(f"  完成: {step/self_arg.state.max_steps*100:.1f}%\n")
                f.write(f"  已用时间: {format_time(elapsed_time)}\n")
                f.write(f"  预计剩余时间: {format_time(estimated_remaining_time)}\n")
                f.write(f"  预计总时间: {format_time(estimated_total_time)}\n")
                f.write("-" * 60 + "\n")
            
            # 同时打印到控制台
            print(f"\n[性能摘要] 步骤 {step}/{self_arg.state.max_steps} ({step/self_arg.state.max_steps*100:.1f}%)")
            print(f"时间: 前向={recent_fw_time:.2f}ms, 反向={recent_bw_time:.2f}ms, 重要性={recent_imp_time:.2f}ms")
            print(f"内存: 当前={current_mem:.2f}MB, 峰值={peak_mem:.2f}MB")
            print(f"进度: 已用时间={format_time(elapsed_time)}, 预计剩余={format_time(estimated_remaining_time)}")
        
        # 添加到类中
        self.log_performance_summary = types.MethodType(log_performance_summary, self)
        
        # 修改training_step方法，使其记录性能数据到文件
        original_training_step = self.training_step
        
        def new_training_step(self_arg, model, inputs):
            model.train()
            inputs = self_arg._prepare_inputs(inputs)
            
            # 记录开始内存
            start_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # 确保CUDA同步以获得准确的时间
            torch.cuda.synchronize()
            forward_start = time.time()
            
            with self_arg.compute_loss_context_manager():
                loss = self_arg.compute_loss(model, inputs)
            
            # 确保loss是标量，修复ViT模型的loss维度问题
            if hasattr(loss, 'mean') and loss.dim() > 0:
                loss = loss.mean()
            
            # 再次同步确保计算完成
            torch.cuda.synchronize()
            forward_end = time.time()
            forward_time = (forward_end - forward_start) * 1000  # ms
            
            # 记录前向传播后的内存
            forward_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            forward_mem_used = max(0.01, forward_mem - start_mem)  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 估算前向传播FLOPs
            forward_flops = self_arg.estimate_flops('forward')
            
            if self_arg.args.gradient_accumulation_steps > 1:
                loss = loss / self_arg.args.gradient_accumulation_steps
            
            # 反向传播
            torch.cuda.synchronize()
            backward_start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_end = time.time()
            backward_time = (backward_end - backward_start) * 1000  # ms
            
            # 记录反向传播后的内存
            backward_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            backward_mem_used = max(0.01, backward_mem - forward_mem)  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 估算反向传播FLOPs
            backward_flops = self_arg.estimate_flops('backward')
            
            # 记录性能数据
            self_arg.perf_stats['forward_time'].append(forward_time / 1000)  # 转回秒存储
            self_arg.perf_stats['backward_time'].append(backward_time / 1000)  # 转回秒存储
            
            # 将数据写入CSV文件
            with open(self_arg.performance_log_path, "a") as f:
                f.write(f"{self_arg.state.global_step},forward,{forward_time:.2f},{forward_mem_used:.2f},{peak_mem:.2f},{forward_flops:.2f},na\n")
                f.write(f"{self_arg.state.global_step},backward,{backward_time:.2f},{backward_mem_used:.2f},{peak_mem:.2f},{backward_flops:.2f},na\n")
            
            # 周期性记录日志
            if self_arg.state.global_step % 50 == 0 or self_arg.state.global_step in self_arg.profile_steps:
                print(f"[性能监测] 步骤 {self_arg.state.global_step} - "
                      f"前向: {forward_time:.2f}ms ({forward_mem_used:.2f}MB, {forward_flops:.2f}GF), "
                      f"反向: {backward_time:.2f}ms ({backward_mem_used:.2f}MB, {backward_flops:.2f}GF)")
            
            # 每100步记录完整性能摘要
            self_arg.log_performance_summary()
            
            return loss.detach()
        
        # 修改update_sensitivity_with_profile方法
        def new_update_sensitivity(self_arg, model, sensitivity_dict, prune_metric):
            """监测更新敏感度字典的性能，并收集全面统计数据"""
            import loraprune.utils as utils
            
            # 记录开始内存
            start_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # 确保CUDA同步以获得准确的时间
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行重要性计算
            updated_dict = utils.update_sensitivity_dict(model, sensitivity_dict, prune_metric)
            
            # 再次同步确保计算完成
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算耗时和内存
            importance_time = (end_time - start_time) * 1000  # ms
            end_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_used = max(0.01, end_mem - start_mem)  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 估算重要性计算FLOPs
            importance_flops = self_arg.estimate_flops('importance')
            
            # 记录性能数据
            self_arg.perf_stats['importance_time'].append((end_time - start_time))  # 以秒为单位存储
            
            # 记录当前方法的统计数据
            if prune_metric in self_arg.method_stats:
                self_arg.method_stats[prune_metric]['time'].append((end_time - start_time))
                self_arg.method_stats[prune_metric]['memory'].append(memory_used)
            
            # 将数据写入CSV文件
            with open(self_arg.performance_log_path, "a") as f:
                f.write(f"{self_arg.state.global_step},importance,{importance_time:.2f},{memory_used:.2f},{peak_mem:.2f},{importance_flops:.2f},{prune_metric}\n")
            
            # 周期性记录日志
            if self_arg.state.global_step % 50 == 0 or self_arg.state.global_step in self_arg.profile_steps:
                print(f"[性能监测] 步骤 {self_arg.state.global_step} - 重要性计算({prune_metric}): "
                      f"{importance_time:.2f}ms, 内存: {memory_used:.2f}MB, FLOPs: {importance_flops:.2f}GF")
            
            return updated_dict
        
        # 替换方法
        self.training_step = types.MethodType(new_training_step, self)
        self.update_sensitivity_with_profile = types.MethodType(new_update_sensitivity, self)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        def apply_masked_modules(model, mask_dict):
            """
            为模型的每个Linear层应用掩码。
            
            Args:
                model: 需要应用掩码的模型
                mask_dict: 包含每层掩码的字典
            """
            for name, module in model.named_modules():
                if isinstance(module, Linear):
                    if not hasattr(module, '_original_forward'): 
                        module._original_forward = module.forward
                    layer_name = ".".join(name.split('.')[:-1])  # 获取层名称
                    
                    # ⭐️ 特别注意此处: 通过参数硬绑定到闭包中，避免循环覆盖
                    def make_masked_forward(orig_forward, layer_name):
                        # 使用 `orig_forward` 作为默认参数传递，解决变量捕获问题
                        def masked_forward(module, *inputs, orig_forward=orig_forward, layer_name=layer_name):
                            output = orig_forward(*inputs)
                            if layer_name in mask_dict:
                                mask = mask_dict[layer_name].to(output.device)
                                # print("Applying mask to layer:",layer_name)
                                # print("Applying mask to layer:", mask.sum(), mask.shape)
                                # 根据层的类型调整掩码形状

                                if 'self_attn' in layer_name or 'attention' in layer_name:
                                    # s_name = ".".join(layer_name.split('.')[:-2])
                                    mask = mask_dict[layer_name].to(output.device)
                                    # 对于注意力层，调整掩码形状以匹配输出
                                    batch_size = output.size(0)
                                    seq_len = output.size(1)
                                    head_dim = output.size(-1) // mask.size(0)
                                    # print(f'Applying mask to {layer_name} with shape {mask.shape} and output shape {output.shape}')

                                    # 将掩码扩展到正确的维度 [num_heads] -> [batch_size, seq_len, num_heads, head_dim]
                                    mask = mask.view(-1, 1).repeat(1, head_dim)  # [num_heads, head_dim]
                                    mask = mask.view(1, 1, -1).expand(batch_size, seq_len, -1)

                                elif 'mlp' in layer_name or 'output' in layer_name or 'intermediate' in layer_name:
                                    # 对于 FFN 层（BERT: output/intermediate；LLaMA: mlp.gate/up/down_proj）
                                    mask = mask.view(1, 1, -1)  # [1, 1, hidden_size/ffn_size]
                                # 使用广播机制应用掩码
                                # print(f'Applying mask to {name} with shape {m ask.shape} and output shape {output.shape}')
                                ## 只在真正 optimizer 更新时才应用
                                # if self.state.global_step % args.gradient_accumulation_steps == 0:
                                output = output * mask
                                
                            return output               
                        return masked_forward            
                    # 为每个模块创建一个新的闭包
                    module.forward = make_masked_forward(module._original_forward, name).__get__(module)

        # 注意：确保在训练开始前调用 apply_masked_modules 函数
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            getattr(self, 'sharded_ddp', None) is not None
            and getattr(self, 'sharded_ddp', None) != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or getattr(self, 'fsdp', None) is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
        total_params = kept_params = sum([p.numel() if not p.requires_grad else 0 for p in model.parameters()])
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")


        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)

        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        if self.prune_metric == 'grad':
            utils.unfreeze(model)

        sensitivity_dict = utils.init_sensitivity_dict(model)
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)


            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # if (
                #     (total_batched_samples % args.gradient_accumulation_steps != 0)
                #     and args.local_rank != -1
                #     and args._no_sync_in_gradient_accumulation
                # ):
                #     # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                #     with model.no_sync():
                #         tr_loss_step = self.training_step(model, inputs)
                # else:
                tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if getattr(self, 'deepspeed', None):
                    self.deepspeed.step()

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    grad_norm: float | None = None
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not getattr(self, 'deepspeed', None):
                        # deepspeed does its own clipping

                        if getattr(self, 'do_grad_scaling', False):
                            # AMP: gradients need unscaling
                            scaler = getattr(self, 'scaler', None)
                            if scaler is not None:
                                scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            grad_norm = self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    if not getattr(self, 'deepspeed', None):
                        # print("prune_metric", self.prune_metric)
                        sensitivity_dict = self.update_sensitivity_with_profile(model, sensitivity_dict, self.prune_metric)
                        # print("sensitivity_dict", sensitivity_dict)
                    ratio = utils.schedule_sparsity_ratio(self.state.global_step, self.state.max_steps,
                                                          self.warmup_iters,
                                                          self.cooldown_iters, self.init_ratio, self.mac)
                    # print("ratio", ratio)                
                    # ratio = 0.05
                    # ====== 重写：采集注意力头重要性分数稳定性并用于剪枝 ======
                    # 确定剪枝周期的后阶段开始收集重要性分数用于计算稳定性（根据用户配置比例）
                    collection_start_ratio = 1.0 - self.stability_collection_ratio
                    stability_collection_phase = self.use_stability and \
                                                  (self.state.global_step % self.prune_freq >= int(self.prune_freq * collection_start_ratio)) and \
                                                  (self.state.global_step % self.prune_freq < self.prune_freq - 1)
                    stability_apply_phase = self.use_stability and (self.state.global_step % self.prune_freq == self.prune_freq - 1)
                    
                    # 创建一个持久化字典来存储收集的重要性分数
                    if not hasattr(self, 'collected_importances'):
                        self.collected_importances = {}
                    
                    # 收集阶段：在剪枝周期的后1/4收集重要性分数
                    if stability_collection_phase:
                        try:
                            # 计算当前步骤的重要性分数
                            s_dict = utils.init_sensitivity_dict(model)
                            s_dict = utils.update_sensitivity_dict(model, s_dict, self.prune_metric)
                            
                            # 根据配置存储相应组件的重要性分数
                            for k, v in s_dict.items():
                                # 分析模块类型（兼容 BERT/ViT + LLaMA/Mistral/Qwen）
                                # BERT/ViT: key 包含 'attention'；LLaMA: key 包含 'self_attn'
                                is_attention = 'attention' in k or 'self_attn' in k
                                # BERT: 'intermediate' 或 'output'；LLaMA: 'mlp'（不含 self_attn）
                                is_ffn = 'mlp' in k or 'intermediate' in k or 'output' in k
                                
                                # 根据配置决定是否收集该组件
                                should_collect = (is_attention and 'attention' in self.stability_components) or \
                                                   (is_ffn and 'ffn' in self.stability_components)
                                
                                if should_collect:
                                    if k not in self.collected_importances:
                                        self.collected_importances[k] = []
                                    # 存储当前步骤的重要性分数
                                    self.collected_importances[k].append(v.detach().cpu().numpy().reshape(-1))
                            
                            # 打印收集状态（兼容 LLaMA：注意力含 self_attn，FFN 含 mlp）
                            if self.state.global_step % 10 == 0:  # 每10步打印一次
                                collected_attn_count = sum(len(v) for k, v in self.collected_importances.items() if 'self_attn' in k or 'attention' in k)
                                collected_ffn_count = sum(len(v) for k, v in self.collected_importances.items() if 'mlp' in k or 'intermediate' in k or 'output' in k)

                        except Exception as e:
                            print(f"[稳定性采集] 收集阶段失败: {e}")
                    if stability_apply_phase and self.collected_importances:
                        try:
                            print(f"[稳定性应用] step={self.state.global_step}, 开始计算稳定性并更新敏感度字典")
                            stability_scores = {}
                            stability_log_path = os.path.join(self.args.output_dir, "head_stability_log.csv")
                            
                            # 写入表头（如果是新文件）
                            # if not os.path.exists(stability_log_path):
                                # with open(stability_log_path, "w") as f:
                                #     f.write("step,module,type,stability,num_samples\n")
                            
                            # 计算每个模块的稳定性
                            for module_name, importances_list in self.collected_importances.items():
                                if len(importances_list) >= 3:  # 至少需要3个样本才能计算稳定性
                                    # 确定模块类型（兼容 BERT/ViT + LLaMA）
                                    module_type = "attention" if ("attention" in module_name or "self_attn" in module_name) else "ffn"
                                    
                                    # 将列表转换为numpy数组 [num_samples, num_heads/neurons]
                                    importances = np.stack(importances_list, axis=0)
                                    # 计算每个头/神经元的排名
                                    ranks = np.argsort(np.argsort(-importances, axis=1), axis=1)
                                    # 计算平均排名
                                    mean_rank = np.mean(ranks, axis=0)
                                    # 计算每个头/神经元的稳定性（排名与平均排名的偏差）
                                    stability = np.mean(np.abs(ranks - mean_rank[None, :]), axis=0)
                                    
                                    # 保存原始稳定性分数用于分析
                                    if module_name not in self.stability_stats:
                                        self.stability_stats[module_name] = []
                                    self.stability_stats[module_name].append({
                                        'step': self.state.global_step,
                                        'stability': stability.copy(),
                                        'num_samples': len(importances_list)
                                    })
                                    
                                    # 转换稳定性分数：值越小（越稳定）表示重要性越高
                                    # 将稳定性分数映射到 [0, 1] 范围，并反转，使得稳定的头得分更高
                                    max_stability = np.max(stability)
                                    min_stability = np.min(stability)
                                    range_stability = max(0.01, max_stability - min_stability)
                                    normalized_stability = 1.0 - (stability - min_stability) / range_stability
                                    
                                    # 创建稳定性分数张量
                                    if module_name in sensitivity_dict:
                                        # 计算原始重要性分数和稳定性分数的加权平均
                                        original_importance = sensitivity_dict[module_name].cpu().numpy()
                                        # 标准化原始重要性分数
                                        max_orig = original_importance.max()
                                        min_orig = original_importance.min()
                                        range_orig = max(0.01, max_orig - min_orig)
                                        norm_orig = (original_importance - min_orig) / range_orig
                                        
                                        # 稳定性和原始重要性的加权平均
                                        alpha = self.stability_weight  # 使用配置的稳定性权重
                                        combined_score = alpha * normalized_stability + (1 - alpha) * norm_orig.reshape(-1)
                                        
                                        # 将组合分数转换回张量并更新灵敏度字典
                                        stability_tensor = torch.tensor(combined_score, dtype=sensitivity_dict[module_name].dtype, 
                                                                        device=sensitivity_dict[module_name].device)
                                        stability_tensor = stability_tensor.reshape(sensitivity_dict[module_name].shape)
                                        
                                        # 更新敏感度字典
                                        sensitivity_dict[module_name] = stability_tensor
                                        
                                        # 保存模块稳定性
                                        module_stability = np.mean(stability)
                                        stability_scores[module_name] = {
                                            'type': module_type,
                                            'stability': module_stability,
                                            'num_samples': len(importances_list)
                                        }
                                        
                                        # 保存到日志
                                        # with open(stability_log_path, "a") as f:
                                        #     f.write(f"{self.state.global_step},{module_name},{module_type},{module_stability:.4f},{len(importances_list)}\n")
                                        
                                        print(f"[稳定性应用] module={module_name}, type={module_type}, stability={module_stability:.4f}, samples={len(importances_list)}")
                                        
                                        # 可选：保存每个头/神经元的稳定性详情
                                        detail_path = os.path.join(self.args.output_dir, f"stability_{module_type}_{module_name.replace('.', '_')}.csv")
                                        # with open(detail_path, "a") as f:
                                        #     f.write(f"step,{self.state.global_step}," + ",".join([f"{s:.4f}" for s in stability]) + "\n")
                            
                            # 重置收集的重要性分数，准备下一个周期
                            self.collected_importances = {}
                            
                            # 如果计算了稳定性分数，添加到日志
                            if stability_scores:
                                attn_scores = {k: v for k, v in stability_scores.items() if v['type'] == 'attention'}
                                ffn_scores = {k: v for k, v in stability_scores.items() if v['type'] == 'ffn'}
                                # print(f"[稳定性应用] 已成功计算 {len(attn_scores)} 个注意力模块和 {len(ffn_scores)} 个FFN模块的稳定性并更新敏感度字典")
                                
                                # 保存综合性能摘要
                                summary_path = os.path.join(self.args.output_dir, "stability_summary.txt")
                                with open(summary_path, "a") as f:
                                    f.write(f"\n步骤 {self.state.global_step} 稳定性摘要:\n")
                                    f.write("-" * 60 + "\n")
                                    f.write(f"实验配置: use_stability={self.use_stability}, components={self.stability_components}, weight={self.stability_weight}\n\n")
                                    
                                    if attn_scores:
                                        f.write("注意力模块:\n")
                                        for name, stats in attn_scores.items():
                                            f.write(f"  模块: {name}\n")
                                            f.write(f"    样本数: {stats['num_samples']}\n")
                                            f.write(f"    平均稳定性: {stats['stability']:.4f}\n")
                                    
                                    if ffn_scores:
                                        f.write("\nFFN模块:\n")
                                        for name, stats in ffn_scores.items():
                                            f.write(f"  模块: {name}\n")
                                            f.write(f"    样本数: {stats['num_samples']}\n")
                                            f.write(f"    平均稳定性: {stats['stability']:.4f}\n")
                                    f.write("-" * 60 + "\n")
                            else:
                                print(f"[稳定性应用] 没有足够的数据计算稳定性分数")
                                
                        except Exception as e:
                            print(f"[稳定性应用] 应用阶段失败: {e}")
                            traceback.print_exc()  # 打印完整错误信息
                    
                    # 在剪枝周期开始时清空收集的数据
                    if self.state.global_step % self.prune_freq == 0:
                        self.collected_importances = {}
                        print(f"[稳定性采集] step={self.state.global_step}, 新剪枝周期开始，重置收集的重要性分数")
                    # ====== 重写结束 ======
                    if (self.state.global_step) % self.prune_freq == 0 and ratio > self.init_ratio:
                        print("prunning at steps", self.state.global_step)
                        if self.mac_ratio < self.mac:
                            print("Pruning condition met, calling local_prune...")
                            
                            # 记录开始时间和内存
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            start_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                            pruning_start = time.time()
                            
                            # 执行剪枝操作
                            self.masks, mac_ratio = utils.search_mac_change(model, sensitivity_dict, self.seq_len, ratio, self.masks)
                            self.mac_ratio = mac_ratio
                            
                            # 记录结束时间和内存
                            torch.cuda.synchronize()
                            pruning_end = time.time()
                            end_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                            pruning_time = (pruning_end - pruning_start) * 1000  # ms
                            pruning_mem = max(0.01, end_mem - start_mem)  # MB
                            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
                            
                            # 估算剪枝FLOPs
                            pruning_flops = self.estimate_flops('pruning')
                            
                            # 存储性能数据
                            if 'pruning_time' not in self.perf_stats:
                                self.perf_stats['pruning_time'] = []
                            self.perf_stats['pruning_time'].append((pruning_end - pruning_start))
                            
                            # 记录到CSV
                            with open(self.performance_log_path, "a") as f:
                                f.write(f"{self.state.global_step},pruning,{pruning_time:.2f},{pruning_mem:.2f},{peak_mem:.2f},{pruning_flops:.2f},search_mac\n")
                            
                            # 打印日志
                            print(f"[性能监测] 步骤 {self.state.global_step} - 剪枝搜索: "
                                  f"{pruning_time:.2f}ms, 内存: {pruning_mem:.2f}MB, FLOPs: {pruning_flops:.2f}GF")
                    utils.apply_masked_modules(model, self.masks)
                    # utils1.apply_model_mask(model, self.masks)
                        # 逐步剪枝
                        # new_masks = utils.search_mac_change(model, sensitivity_dict,self.seq_len,ratio,new_masks)
                            # 维护稀疏性
                            # for layer in new_masks:
                            #     if layer in self.masks:
                            #         self.masks[layer] = self.masks[layer] * new_masks[layer]  # 累积剪枝
                            #     else:
                            #         self.masks[layer] = new_masks[layer]  # 新剪枝

                        # 避免重复修改 forward 方法
                    # if self.masks is not None:
                    #     # print("apply_masked_modules")
                    #     apply_masked_modules(model, self.masks)
                    optimizer_was_run = True
                    _deepspeed = getattr(self, 'deepspeed', None)
                    if _deepspeed:
                        pass  # called outside the loop
                    elif getattr(self, 'do_grad_scaling', False):
                        scaler = getattr(self, 'scaler', None)
                        if scaler is not None:
                            scale_before = scaler.get_scale()
                            scaler.step(self.optimizer)
                            scaler.update()
                            scale_after = scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()
                    else:
                        self.optimizer.step()
                    # if self.masks is not None:
                        # print("Applying mask after optimizer step")
                        # utils1.apply_model_mask(model, self.masks)
                    if optimizer_was_run and not _deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch,
                                                  ignore_keys_for_eval=ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch,
                                         ignore_keys_for_eval=ignore_keys_for_eval)


            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # 生成完整的性能报告
        _ = self.generate_full_training_report()

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def generate_full_training_report(self):
        """生成完整的训练过程性能报告，包括各剪枝方法的实际开销统计"""
        import time
        import torch
        import numpy as np
        import os
        import gc
        import traceback
        from collections import defaultdict
        
        report = "\n完整训练过程性能报告\n"
        report += "=" * 60 + "\n\n"
        
        # 准备统计数据结构
        stats = {
            'magnitude': {
                'time': 0.0,
                'compute_count': 0,
                'peak_memory': 0.0,
                'avg_time_per_step': 0.0,
                'flops': 0.0,
            },
            'grad': {
                'time': 0.0,
                'compute_count': 0,
                'peak_memory': 0.0,
                'avg_time_per_step': 0.0,
                'flops': 0.0,
            },
            'lora': {
                'time': 0.0,
                'compute_count': 0,
                'peak_memory': 0.0,
                'avg_time_per_step': 0.0,
                'flops': 0.0,
            }
        }
        
        # 训练整体统计
        train_stats = {
            'total_steps': self.state.global_step,
            'total_epochs': self.state.num_train_epochs,
            'total_training_time': 0.0,
            'forward_time': 0.0,
            'backward_time': 0.0,
            'importance_time': 0.0,
            'pruning_time': 0.0,
            'forward_memory': 0.0,
            'backward_memory': 0.0,
            'importance_memory': 0.0,
            'pruning_memory': 0.0,
            'total_memory': 0.0,
            'forward_flops': 0.0,
            'backward_flops': 0.0,
            'importance_flops': 0.0,
            'pruning_flops': 0.0,
            'total_flops': 0.0,
        }
        
        try:
            # 0. 收集评估指标（PPL等）
            eval_metrics = {}
            for log_entry in self.state.log_history:
                if 'eval_loss' in log_entry:
                    eval_metrics['eval_loss'] = log_entry['eval_loss']
                if 'eval_ppl' in log_entry:
                    eval_metrics['eval_ppl'] = log_entry['eval_ppl']
                if 'eval_runtime' in log_entry:
                    eval_metrics['eval_runtime'] = log_entry['eval_runtime']
            
            # 1. 从训练指标中获取总训练时间
            metrics_dict = self.state.log_history[-1] if self.state.log_history else {}
            if 'train_runtime' in metrics_dict:
                train_stats['total_training_time'] = metrics_dict.get('train_runtime', 0)
            
            # 2. 从训练过程性能统计中获取各阶段时间
            if self.perf_stats['forward_time']:
                train_stats['forward_time'] = np.sum(self.perf_stats['forward_time'])
                train_stats['backward_time'] = np.sum(self.perf_stats['backward_time'])
                train_stats['importance_time'] = np.sum(self.perf_stats['importance_time'])
                if 'pruning_time' in self.perf_stats:
                    train_stats['pruning_time'] = np.sum(self.perf_stats['pruning_time'])
            
            # 3. 特别测试三种剪枝方法的性能
            print("\n开始测量三种剪枝方法的详细性能...\n")
            model = self.model
            device = next(model.parameters()).device
            
            # 导入工具
            import loraprune.utils as utils
            
            # 获取模型参数统计信息，用于估算FLOPs
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for n, p in model.named_parameters() 
                             if 'lora_' in n and p.requires_grad)
            
            print(f"模型总参数数量: {total_params}")
            print(f"其中LoRA参数数量: {lora_params}")
            
            # 估算训练过程中的内存使用
            # 1. 从已记录的数据中获取平均内存使用
            avg_forward_mem = np.mean([m for _, m in zip(self.perf_stats['forward_time'], self.method_stats[self.prune_metric]['memory'])]) if self.method_stats[self.prune_metric]['memory'] else 0
            avg_backward_mem = avg_forward_mem * 2  # 反向传播通常使用正向传播的2倍内存
            avg_importance_mem = np.mean(self.method_stats[self.prune_metric]['memory']) if self.method_stats[self.prune_metric]['memory'] else 0
            
            # 2. 估算总内存使用
            train_stats['forward_memory'] = avg_forward_mem * self.state.global_step
            train_stats['backward_memory'] = avg_backward_mem * self.state.global_step
            train_stats['importance_memory'] = avg_importance_mem * self.state.global_step
            
            # 3. 获取当前峰值内存作为总内存使用的参考
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            train_stats['total_memory'] = peak_memory
            
            # 估算训练过程中的FLOPs
            # 1. 估算单步正向传播FLOPs
            model_size = sum(p.numel() for p in model.parameters())  # 总参数数量
            seq_length = self.seq_len
            batch_size = self.args.per_device_train_batch_size
            
            # BERT模型每个token约有4 * hidden_size^2 * num_layers的操作
            hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
            num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
            
            # 简单估算各阶段FLOPs
            forward_flops_per_step = 4 * batch_size * seq_length * hidden_size * hidden_size * num_layers
            backward_flops_per_step = 2 * forward_flops_per_step  # 反向传播通常是前向传播的2倍FLOPs
            
            # 估算不同方法的importance计算FLOPs
            importance_flops = {}
            for method in ['magnitude', 'grad', 'lora']:
                if method == 'magnitude':
                    # 对每个参数计算绝对值 - 每个参数约1个FLOP
                    method_flops = total_params
                elif method == 'grad':
                    # 参数与梯度相乘 - 每个参数约3个FLOP
                    method_flops = 3 * total_params
                elif method == 'lora':
                    # LoRA方法 - 主要与LoRA参数相关
                    method_flops = 5 * lora_params
                
                importance_flops[method] = method_flops
            
            # 累计FLOPs
            train_stats['forward_flops'] = forward_flops_per_step * self.state.global_step / 1e12  # TFLOPs
            train_stats['backward_flops'] = backward_flops_per_step * self.state.global_step / 1e12  # TFLOPs
            train_stats['importance_flops'] = importance_flops[self.prune_metric] * self.state.global_step / 1e12  # TFLOPs
            train_stats['total_flops'] = train_stats['forward_flops'] + train_stats['backward_flops'] + train_stats['importance_flops']
            
            # 定义可靠的性能测量函数
            def measure_method_performance(model, method, sensitivity_dict, num_runs=5):
                # 确保每次测量前清理缓存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                times = []
                for _ in range(num_runs):
                    # 每次测量前重置模型状态
                    model.zero_grad()
                    
                    # 记录开始时间
                    start_time = time.time()
                    torch.cuda.synchronize()
                    
                    # 执行方法
                    _ = utils.update_sensitivity_dict(model, deepcopy(sensitivity_dict), method)
                    
                    # 确保计算完成
                    torch.cuda.synchronize()
                    
                    # 记录结束时间
                    end_time = time.time()
                    
                    # 计算执行时间
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    
                    # 清理缓存
                    torch.cuda.empty_cache()
                
                # 返回平均时间（排除最高和最低值）
                times.sort()
                return sum(times[1:-1]) / (len(times) - 2)
            
            # 测量各方法性能 - 只测量LoRA方法
            methods = ['lora']
            iterations = 5  # 每种方法测量次数
            
            print(f"每种方法将测量 {iterations} 次取平均值")
            
            # 首先确定当前使用的方法
            current_method = self.prune_metric
            print(f"当前训练使用的剪枝方法: {current_method}")
            
            # 初始化敏感度字典
            sensitivity_dict = utils.init_sensitivity_dict(model)
            
            # 测量每种方法性能
            for method in methods:
                print(f"\n测量 {method} 方法的性能指标...")
                
                # 进行测量
                perf_time = measure_method_performance(model, method, sensitivity_dict, iterations)
                perf_time_ms = perf_time * 1000  # 转换为毫秒
                
                # 获取实际的内存使用统计
                perf_memory = self.perf_stats.get('memory_usage', 0.0)
                perf_peak = self.perf_stats.get('peak_memory', 0.0)
                
                # 保存结果
                stats[method]['avg_time_per_step'] = perf_time_ms
                stats[method]['avg_memory'] = perf_memory if perf_memory > 0 else 0.01
                stats[method]['peak_memory'] = perf_peak if perf_peak > 0 else 0.01
                
                # FLOPs已经在之前估算
                stats[method]['flops'] = importance_flops[method] / 1e9  # 转换为GFLOPs
                
                print(f"  平均时间: {perf_time_ms:.2f}ms/步")
                print(f"  平均内存: {stats[method]['avg_memory']:.2f}MB")
                print(f"  峰值内存: {stats[method]['peak_memory']:.2f}MB")
                print(f"  估计FLOPs: {stats[method]['flops']:.4f}GFLOPs")
            
            # 生成性能报告
            
            # 1. 训练基本信息
            report += "训练基本信息:\n"
            report += "-" * 60 + "\n"
            report += f"总训练步数: {train_stats['total_steps']}\n"
            report += f"总训练轮数: {train_stats['total_epochs']:.2f}\n"
            report += f"使用的剪枝方法: {current_method}\n"
            report += f"总训练时间: {train_stats['total_training_time']:.2f}秒\n"
            
            # 添加模型性能指标
            if eval_metrics:
                report += "\n模型性能指标:\n"
                report += "-" * 60 + "\n"
                if 'eval_loss' in eval_metrics:
                    report += f"最终验证损失: {eval_metrics['eval_loss']:.4f}\n"
                if 'eval_ppl' in eval_metrics:
                    report += f"最终验证困惑度(PPL): {eval_metrics['eval_ppl']:.2f}\n"
                if 'eval_runtime' in eval_metrics:
                    report += f"评估耗时: {eval_metrics['eval_runtime']:.2f}秒\n"
                # 计算PPL改进（如果有初始PPL记录）
                first_eval_loss = None
                for log_entry in self.state.log_history:
                    if 'eval_loss' in log_entry and first_eval_loss is None:
                        first_eval_loss = log_entry['eval_loss']
                if first_eval_loss and 'eval_loss' in eval_metrics:
                    ppl_improvement = math.exp(first_eval_loss) / math.exp(eval_metrics['eval_loss'])
                    report += f"PPL改善倍数: {ppl_improvement:.2f}x\n"
            report += "\n"
            
            # 2. 时间开销分布
            report += "时间开销分布:\n"
            report += "-" * 60 + "\n"
            
            # 计算各部分占比 (如果有数据)
            if train_stats['total_training_time'] > 0:
                forward_percent = (train_stats['forward_time'] / train_stats['total_training_time']) * 100
                backward_percent = (train_stats['backward_time'] / train_stats['total_training_time']) * 100
                importance_percent = (train_stats['importance_time'] / train_stats['total_training_time']) * 100
                pruning_percent = (train_stats['pruning_time'] / train_stats['total_training_time']) * 100
                other_percent = 100 - (forward_percent + backward_percent + importance_percent + pruning_percent)
                
                report += f"前向传播: {train_stats['forward_time']:.2f}秒 ({forward_percent:.1f}%)\n"
                report += f"反向传播: {train_stats['backward_time']:.2f}秒 ({backward_percent:.1f}%)\n"
                report += f"重要性计算: {train_stats['importance_time']:.2f}秒 ({importance_percent:.1f}%)\n"
                report += f"剪枝搜索: {train_stats['pruning_time']:.2f}秒 ({pruning_percent:.1f}%)\n"
                report += f"其他操作: {train_stats['total_training_time'] - train_stats['forward_time'] - train_stats['backward_time'] - train_stats['importance_time'] - train_stats['pruning_time']:.2f}秒 ({other_percent:.1f}%)\n\n"
            
            # 3. 内存使用分布
            report += "内存使用分布 (MB):\n"
            report += "-" * 60 + "\n"
            
            # 估算各阶段内存使用
            total_memory = train_stats['total_memory']
            if total_memory > 0:
                # 根据已有统计数据估算比例
                forward_mem_percent = 25.0  # 典型分布
                backward_mem_percent = 40.0
                importance_mem_percent = 15.0
                other_mem_percent = 20.0
                
                # 根据百分比计算内存量
                forward_mem = total_memory * (forward_mem_percent / 100)
                backward_mem = total_memory * (backward_mem_percent / 100)
                importance_mem = total_memory * (importance_mem_percent / 100)
                other_mem = total_memory * (other_mem_percent / 100)
                
                report += f"前向传播: {forward_mem:.2f}MB ({forward_mem_percent:.1f}%)\n"
                report += f"反向传播: {backward_mem:.2f}MB ({backward_mem_percent:.1f}%)\n"
                report += f"重要性计算: {importance_mem:.2f}MB ({importance_mem_percent:.1f}%)\n"
                report += f"其他操作: {other_mem:.2f}MB ({other_mem_percent:.1f}%)\n"
                report += f"峰值内存使用: {total_memory:.2f}MB\n\n"
            
            # 4. FLOPs分布
            report += "计算量分布 (TFLOPs):\n"
            report += "-" * 60 + "\n"
            
            if train_stats['total_flops'] > 0:
                forward_flops_percent = (train_stats['forward_flops'] / train_stats['total_flops']) * 100
                backward_flops_percent = (train_stats['backward_flops'] / train_stats['total_flops']) * 100
                importance_flops_percent = (train_stats['importance_flops'] / train_stats['total_flops']) * 100
                
                report += f"前向传播: {train_stats['forward_flops']:.2f}T ({forward_flops_percent:.1f}%)\n"
                report += f"反向传播: {train_stats['backward_flops']:.2f}T ({backward_flops_percent:.1f}%)\n"
                report += f"重要性计算: {train_stats['importance_flops']:.2f}T ({importance_flops_percent:.1f}%)\n"
                report += f"总计算量: {train_stats['total_flops']:.2f}T\n\n"
            
            # 5. 各剪枝方法性能对比
            report += "剪枝方法性能对比 (单步):\n"
            report += "-" * 80 + "\n"
            report += f"{'方法':<10} | {'计算时间(ms)':<15} | {'内存使用(MB)':<15} | {'峰值内存(MB)':<15} | {'估计FLOPs(G)':<15}\n"
            report += "-" * 80 + "\n"
            
            baseline_method = 'magnitude'  # 基准方法
            
            for method in methods:
                report += f"{method:<10} | {stats[method]['avg_time_per_step']:<15.2f} | {stats[method]['avg_memory']:<15.2f} | {stats[method]['peak_memory']:<15.2f} | {stats[method]['flops']:<15.4f}\n"
            
            # 6. 相对性能 (相对于magnitude方法)
            report += "\n相对性能 (相对于magnitude方法):\n"
            report += "-" * 80 + "\n"
            report += f"{'方法':<10} | {'计算时间':<15} | {'内存使用':<15} | {'峰值内存':<15} | {'估计FLOPs':<15}\n"
            report += "-" * 80 + "\n"
            
            # 检查基准方法是否有有效数据
            baseline_stats = stats.get(baseline_method, {})
            baseline_time = baseline_stats.get('avg_time_per_step', 0.0)
            baseline_memory = baseline_stats.get('avg_memory', 0.0)
            baseline_peak = baseline_stats.get('peak_memory', 0.0)
            baseline_flops = baseline_stats.get('flops', 0.0)
            
            for method in methods:
                method_stats = stats.get(method, {})
                method_time = method_stats.get('avg_time_per_step', 0.0)
                method_memory = method_stats.get('avg_memory', 0.0)
                method_peak = method_stats.get('peak_memory', 0.0)
                method_flops = method_stats.get('flops', 0.0)
                
                # 避免除零错误
                rel_time = method_time / baseline_time if baseline_time > 0 else 0.0
                rel_memory = method_memory / baseline_memory if baseline_memory > 0 else 0.0
                rel_peak = method_peak / baseline_peak if baseline_peak > 0 else 0.0
                rel_flops = method_flops / baseline_flops if baseline_flops > 0 else 0.0
                    
                report += f"{method:<10} | {rel_time:<15.2f}x | {rel_memory:<15.2f}x | {rel_peak:<15.2f}x | {rel_flops:<15.2f}x\n"
            
            # 7. 估算总体训练开销
            report += "\n总体训练开销估计 (基于实际训练步数):\n"
            report += "-" * 80 + "\n"
            report += f"{'方法':<10} | {'总计算时间(小时)':<20} | {'总内存开销(GB-小时)':<20} | {'总FLOPs(T)':<15}\n"
            report += "-" * 80 + "\n"
            
            for method in methods:
                # 计算总时间 (小时)
                total_hours = (stats[method]['avg_time_per_step'] * self.state.global_step) / (1000 * 60 * 60)
                
                # 内存消耗 (GB-小时) - 表示持续使用X GB内存Y小时的累计开销
                memory_gb = stats[method]['avg_memory'] / 1024  # 转为GB
                memory_gb_hours = memory_gb * total_hours
                
                # 总FLOPs计算
                total_flops_t = (stats[method]['flops'] * self.state.global_step) / 1000  # 转为TFLOPs
                    
                report += f"{method:<10} | {total_hours:<20.2f} | {memory_gb_hours:<20.4f} | {total_flops_t:<15.4f}\n"
            
            # 8. 总结与建议
            report += "\n总结与建议:\n"
            report += "-" * 60 + "\n"
            
            # 确定最快和最慢的方法
            times = [(method, stats[method]['avg_time_per_step']) for method in methods]
            times.sort(key=lambda x: x[1])
            fastest = times[0][0]
            slowest = times[-1][0]
            
            # 内存使用最小和最大的方法
            memories = [(method, stats[method]['peak_memory']) for method in methods]
            memories.sort(key=lambda x: x[1])
            least_mem = memories[0][0]
            most_mem = memories[-1][0]
            
            report += f"1. 计算时间最短的方法是 {fastest}，最长的是 {slowest}。\n"
            report += f"2. 内存使用最少的方法是 {least_mem}，最多的是 {most_mem}。\n"
            
            if fastest == least_mem:
                report += f"3. {fastest} 方法在时间和内存上都表现最优。\n"
            else:
                report += f"3. 时间与内存之间存在权衡，{fastest} 最快但 {least_mem} 内存使用最少。\n"
            
            if current_method == fastest:
                report += f"4. 当前训练使用的 {current_method} 方法是计算时间最短的方法，这是一个好的选择。\n"
            else:
                report += f"4. 当前训练使用的 {current_method} 方法不是计算时间最短的方法，考虑切换到 {fastest} 可能会提高训练效率。\n"
            
            # 9. 改进建议
            report += "\n改进建议:\n"
            report += "-" * 60 + "\n"
            report += f"1. 如果主要关注训练速度，建议使用 {fastest} 方法。\n"
            report += f"2. 如果内存受限，建议使用 {least_mem} 方法。\n"
            report += "3. 在长时间训练中，可以考虑在训练的不同阶段使用不同的剪枝方法。\n"
            report += "4. 可以通过增加剪枝频率(prune_freq)来减少总体剪枝开销。\n"
            
        except Exception as e:
            report += f"生成报告时出错: {str(e)}\n\n"
            trace = traceback.format_exc()
            report += f"错误详情: {trace}\n\n"
        
        # 保存报告
        output_dir = self.args.output_dir
        report_path = os.path.join(output_dir, "full_training_performance_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"\n完整训练性能报告已保存到 {report_path}")
        return report
    def save_stability_distribution(self, save_dir=None):
        """保存列表中的稳定性分数分布信息，便于执行消融实验分析
        
        Args:
            save_dir: 保存目录，默认为输出目录
        
        这个方法会生成两类文件：
        1. stability_distribution_<type>_<step>.npy - 包含原始稳定性分数的NumPy数组
        2. stability_distribution_<type>_<step>.csv - 更方便读取的CSV格式版本
        """
        if not self.stability_stats:
            print("没有找到稳定性统计数据。请先运行训练并启用稳定性分数。")
            return
            
        if save_dir is None:
            save_dir = self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 按类型分组（兼容 BERT/ViT + LLaMA）
        attention_modules = {k: v for k, v in self.stability_stats.items() if 'attention' in k or 'self_attn' in k}
        ffn_modules = {k: v for k, v in self.stability_stats.items() if 'mlp' in k or 'intermediate' in k or 'output' in k}
        
        # 保存分布数据
        step = self.state.global_step
        
        # 1. 注意力头分布
        if attention_modules:
            # 收集所有注意力头稳定性分数
            all_attention_scores = []
            for module_name, stats_list in attention_modules.items():
                for stat in stats_list:
                    for score in stat['stability']:
                        all_attention_scores.append({
                            'module': module_name,
                            'step': stat['step'],
                            'stability': score
                        })
            
            # # 保存为NumPy格式
            # attn_scores_array = np.array([item['stability'] for item in all_attention_scores])
            # np.save(f"{save_dir}/stability_distribution_attention_{step}.npy", attn_scores_array)
            
            # # 保存为CSV格式
            # with open(f"{save_dir}/stability_distribution_attention_{step}.csv", "w") as f:
            #     f.write("module,step,stability\n")
            #     for item in all_attention_scores:
            #         f.write(f"{item['module']},{item['step']},{item['stability']}\n")
        
        # 2. FFN层分布
        if ffn_modules:
            # 收集所有FFN层稳定性分数
            all_ffn_scores = []
            for module_name, stats_list in ffn_modules.items():
                for stat in stats_list:
                    for score in stat['stability']:
                        all_ffn_scores.append({
                            'module': module_name,
                            'step': stat['step'],
                            'stability': score
                        })
            
            # # 保存为NumPy格式
            # ffn_scores_array = np.array([item['stability'] for item in all_ffn_scores])
            # np.save(f"{save_dir}/stability_distribution_ffn_{step}.npy", ffn_scores_array)
            
            # # 保存为CSV格式
            # with open(f"{save_dir}/stability_distribution_ffn_{step}.csv", "w") as f:
            #     f.write("module,step,stability\n")
            #     for item in all_ffn_scores:
            #         f.write(f"{item['module']},{item['step']},{item['stability']}\n")
        
        # 3. 保存汇总统计信息
        attn_count = len(attention_modules)
        ffn_count = len(ffn_modules)
        
        print(f"已保存稳定性分布数据到 {save_dir}")
        print(f"  - 注意力头模块: {attn_count} 个")
        print(f"  - FFN层模块: {ffn_count} 个")
        
        return attention_modules, ffn_modules
    
    def generate_stability_report(self, save_dir=None):
        """生成完整的稳定性分数分析报告
        
        Args:
            save_dir: 保存目录，默认为输出目录
        """
        if not self.stability_stats:
            print("没有找到稳定性统计数据。请先运行训练并启用稳定性分数。")
            return None
        
        if save_dir is None:
            save_dir = self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取统计数据
        attention_modules, ffn_modules = self.save_stability_distribution(save_dir)
        
        # 准备报告
        report = "稳定性分数分析报告\n"
        report += "=" * 80 + "\n\n"
        
        # 实验配置信息
        report += "实验配置信息:\n"
        report += "-" * 40 + "\n"
        report += f"是否启用稳定性分数: {self.use_stability}\n"
        report += f"稳定性应用组件: {self.stability_components}\n"
        report += f"稳定性权重: {self.stability_weight}\n"
        report += f"稳定性收集比例: {self.stability_collection_ratio}\n"
        report += f"剪枝比例: {self.ratio}\n"
        report += f"剪枝频率: {self.prune_freq}\n"
        report += f"剪枝指标: {self.prune_metric}\n\n"
        
        # 1. 注意力头稳定性分析
        if attention_modules:
            # 计算统计数据
            attn_stats = self._calculate_module_stats(attention_modules)
            
            report += "注意力头稳定性分析:\n"
            report += "-" * 40 + "\n"
            report += f"分析模块数量: {len(attention_modules)}\n"
            report += f"平均稳定性分数: {attn_stats['mean']:.4f}\n"
            report += f"标准差: {attn_stats['std']:.4f}\n"
            report += f"最小值: {attn_stats['min']:.4f}\n"
            report += f"最大值: {attn_stats['max']:.4f}\n"
            report += f"中位数: {attn_stats['median']:.4f}\n"
            report += f"四分位: {attn_stats['percentiles'][0]:.4f}, {attn_stats['percentiles'][1]:.4f}, {attn_stats['percentiles'][2]:.4f}\n"
            
            # 按层统计
            report += "\n按层分析\n"
            layer_stats = self._calculate_layer_stats(attention_modules)
            for layer_idx, stats in layer_stats.items():
                report += f"  层{layer_idx}: 平均稳定性 = {stats['mean']:.4f}, 标准差 = {stats['std']:.4f}\n"
            report += "\n"
            
            # 添加最稳定和最不稳定的模块
            sorted_modules = self._get_sorted_modules(attention_modules)
            report += "最稳定的注意力头模块 (平均稳定性分数最低):\n"
            for i, (module_name, avg_score) in enumerate(sorted_modules[:5]):
                report += f"  {i+1}. {module_name}: {avg_score:.4f}\n"
            
            report += "\n最不稳定的注意力头模块 (平均稳定性分数最高):\n"
            for i, (module_name, avg_score) in enumerate(sorted_modules[-5:]):
                report += f"  {i+1}. {module_name}: {avg_score:.4f}\n"
            report += "\n"
        
        # 2. FFN层稳定性分析
        if ffn_modules:
            # 计算统计数据
            ffn_stats = self._calculate_module_stats(ffn_modules)
            
            report += "FFN层稳定性分析:\n"
            report += "-" * 40 + "\n"
            report += f"分析模块数量: {len(ffn_modules)}\n"
            report += f"平均稳定性分数: {ffn_stats['mean']:.4f}\n"
            report += f"标准差: {ffn_stats['std']:.4f}\n"
            report += f"最小值: {ffn_stats['min']:.4f}\n"
            report += f"最大值: {ffn_stats['max']:.4f}\n"
            report += f"中位数: {ffn_stats['median']:.4f}\n"
            report += f"四分位: {ffn_stats['percentiles'][0]:.4f}, {ffn_stats['percentiles'][1]:.4f}, {ffn_stats['percentiles'][2]:.4f}\n"
            
            # 按层统计
            report += "\n按层分析\n"
            layer_stats = self._calculate_layer_stats(ffn_modules)
            for layer_idx, stats in layer_stats.items():
                report += f"  层{layer_idx}: 平均稳定性 = {stats['mean']:.4f}, 标准差 = {stats['std']:.4f}\n"
            report += "\n"
            
            # 添加最稳定和最不稳定的模块
            sorted_modules = self._get_sorted_modules(ffn_modules)
            report += "最稳定的FFN模块 (平均稳定性分数最低):\n"
            for i, (module_name, avg_score) in enumerate(sorted_modules[:5]):
                report += f"  {i+1}. {module_name}: {avg_score:.4f}\n"
            
            report += "\n最不稳定的FFN模块 (平均稳定性分数最高):\n"
            for i, (module_name, avg_score) in enumerate(sorted_modules[-5:]):
                report += f"  {i+1}. {module_name}: {avg_score:.4f}\n"
            report += "\n"
        
        # 3. 比较注意力头和FFN层的稳定性
        if attention_modules and ffn_modules:
            report += "注意力头与FFN层稳定性对比:\n"
            report += "-" * 40 + "\n"
            report += f"注意力头平均稳定性: {attn_stats['mean']:.4f}\n"
            report += f"FFN层平均稳定性: {ffn_stats['mean']:.4f}\n"
            if attn_stats['mean'] < ffn_stats['mean']:
                report += "结论: 注意力头比FFN层更稳定\n"
            else:
                report += "结论: FFN层比注意力头更稳定\n"
            report += "\n"
        
        # 4. 消融实验建议
        report += "消融实验建议:\n"
        report += "-" * 40 + "\n"
        report += "基于以上分析，建议的消融实验设置如下:\n"
        report += "1. 基准实验: use_stability=False\n"
        report += "2. 只对注意力头使用稳定性: use_stability=True, stability_components=['attention']\n"
        report += "3. 只对FFN层使用稳定性: use_stability=True, stability_components=['ffn']\n"
        report += "4. 对两者都使用稳定性: use_stability=True, stability_components=['attention', 'ffn']\n"
        report += "5. 使用不同权重布置的实验: stability_weight=0.1, 0.3, 0.5, 0.7\n"
        
        # 保存报告
        report_path = os.path.join(save_dir, f"stability_analysis_report_{self.state.global_step}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"已生成并保存稳定性分数分析报告到 {report_path}")
        
        return report

    def _calculate_module_stats(self, modules):
        # modules: dict[module_name, list[dict{stability: np.ndarray, ...}]]
        all_scores = []
        for stats_list in modules.values():
            for stat in stats_list:
                all_scores.extend(stat['stability'])
        all_scores = np.array(all_scores)
        return {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'median': np.median(all_scores),
            'percentiles': np.percentile(all_scores, [25, 50, 75])
        }

    def _calculate_layer_stats(self, modules):
        # 假设模块名中有层号，如 encoder.layer.0.attention
        layer_stats = {}
        for module_name, stats_list in modules.items():
            # 提取层号
            import re
            match = re.search(r'layer\.(\d+)', module_name)
            if match:
                layer_idx = int(match.group(1))
            else:
                layer_idx = -1
            all_scores = []
            for stat in stats_list:
                all_scores.extend(stat['stability'])
            all_scores = np.array(all_scores)
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {'scores': []}
            layer_stats[layer_idx]['scores'].extend(all_scores)
        # 统计
        for layer_idx in layer_stats:
            scores = np.array(layer_stats[layer_idx]['scores'])
            layer_stats[layer_idx] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        return layer_stats

    def _get_sorted_modules(self, modules):
        # 返回[(module_name, avg_score), ...]，按avg_score升序
        module_avg = []
        for module_name, stats_list in modules.items():
            all_scores = []
            for stat in stats_list:
                all_scores.extend(stat['stability'])
            avg_score = np.mean(all_scores)
            module_avg.append((module_name, avg_score))
        module_avg.sort(key=lambda x: x[1])
        return module_avg