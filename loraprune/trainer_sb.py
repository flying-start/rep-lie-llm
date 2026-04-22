from transformers.trainer import *
import utils as utils
import utils1 as utils1
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

class StabilityLoRAPruneTrainer(Trainer):
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
        
        # 性能监测相关属性
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
        self.profile_steps = list(range(10, 500, 50))  # 采样性能数据
        
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """训练主循环，包含稳定性分数计算和应用"""
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        def apply_masked_modules(model, mask_dict):
            """为模型的每个Linear层应用掩码"""
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
                                # 根据层的类型调整掩码形状
                                if 'attention' in layer_name:
                                    mask = mask_dict[layer_name].to(output.device)
                                    # 对于注意力层，调整掩码形状以匹配输出
                                    batch_size = output.size(0)
                                    seq_len = output.size(1)
                                    head_dim = output.size(-1) // mask.size(0)
                                    
                                    # 将掩码扩展到正确的维度
                                    mask = mask.view(-1, 1).repeat(1, head_dim)  # [num_heads, head_dim]
                                    mask = mask.view(1, 1, -1).expand(batch_size, seq_len, -1)
                                        
                                elif 'output' in layer_name or 'intermediate' in layer_name:
                                    # 对于输出层和中间层，调整掩码形状
                                    mask = mask.view(1, 1, -1)
                                # 使用广播机制应用掩码
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

        # 激活梯度检查点（如果需要）
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # 创建优化器和学习率调度器
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 初始化训练状态
        self.state = TrainerState()
        self.state.epoch = 0
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # 设置损失追踪
        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        # 开始训练循环
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        if self.prune_metric == 'grad':
            utils.unfreeze(model)

        sensitivity_dict = utils.init_sensitivity_dict(model)
        
        # 主训练循环
        for epoch in range(0, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            steps_in_epoch = (
                len(train_dataloader) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(train_dataloader):
                total_batched_samples += 1
                
                # 检查当前步是否是梯度累积的起始步
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # 执行训练步骤
                tr_loss_step = self.training_step(model, inputs)
                tr_loss += tr_loss_step

                # 检查是否需要执行优化器步骤
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # 梯度裁剪
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        if self.do_grad_scaling:
                            # AMP: 需要解除梯度缩放
                            self.scaler.unscale_(self.optimizer)
                        
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # 一些优化器有特定的梯度裁剪方式
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        else:
                            # 否则使用常规裁剪
                            nn.utils.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # ====== 稳定性分数计算和应用逻辑 ======
                    # 确定剪枝周期的后阶段开始收集重要性分数用于计算稳定性（根据用户配置比例）
                    collection_start_ratio = 1.0 - self.stability_collection_ratio
                    stability_collection_phase = self.use_stability and \
                                                  (self.state.global_step % self.prune_freq >= int(self.prune_freq * collection_start_ratio)) and \
                                                  (self.state.global_step % self.prune_freq < self.prune_freq - 1)
                    stability_apply_phase = self.use_stability and (self.state.global_step % self.prune_freq == self.prune_freq - 1)
                     
                    # 收集阶段：在剪枝周期的后1/4收集重要性分数
                    if stability_collection_phase:
                        try:
                            # 计算当前步骤的重要性分数
                            s_dict = utils.init_sensitivity_dict(model)
                            s_dict = utils.update_sensitivity_dict(model, s_dict, self.prune_metric)
                             
                            # 根据配置存储相应组件的重要性分数
                            for k, v in s_dict.items():
                                # 分析模块类型
                                is_attention = 'attention' in k
                                is_ffn = 'intermediate' in k or 'output' in k  # FFN层通常包含intermediate和output
                                 
                                # 根据配置决定是否收集该组件
                                should_collect = (is_attention and 'attention' in self.stability_components) or \
                                                  (is_ffn and 'ffn' in self.stability_components)
                                 
                                if should_collect:
                                    if k not in self.collected_importances:
                                        self.collected_importances[k] = []
                                    # 存储当前步骤的重要性分数
                                    self.collected_importances[k].append(v.detach().cpu().numpy().reshape(-1))
                             
                            # 打印收集状态
                            if self.state.global_step % 10 == 0:  # 每10步打印一次
                                collected_attn_count = sum(len(v) for k, v in self.collected_importances.items() if 'attention' in k)
                                collected_ffn_count = sum(len(v) for k, v in self.collected_importances.items() if 'intermediate' in k or 'output' in k)
                                print(f"[稳定性采集] step={self.state.global_step}, 已收集 {collected_attn_count} 条注意力头和 {collected_ffn_count} 条FFN层重要性分数记录")
                         
                        except Exception as e:
                            print(f"[稳定性采集] 收集阶段失败: {e}")
                     
                    # 应用阶段：在剪枝周期的最后一步计算稳定性并应用于下一次剪枝f_calculate_module_stats
                    if stability_apply_phase and self.collected_importances:
                        try:
                            print(f"[稳定性应用] step={self.state.global_step}, 开始计算稳定性并更新敏感度字典")
                            stability_scores = {}
                            stability_log_path = os.path.join(self.args.output_dir, "head_stability_log.csv")
                             
                            # 写入表头（如果是新文件）
                            if not os.path.exists(stability_log_path):
                                with open(stability_log_path, "w") as f:
                                    f.write("step,module,type,stability,num_samples\n")
                             
                            # 计算每个模块的稳定性
                            for module_name, importances_list in self.collected_importances.items():
                                if len(importances_list) >= 3:  # 至少需要3个样本才能计算稳定性
                                    # 确定模块类型
                                    module_type = "attention" if "attention" in module_name else "ffn"
                                     
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
                                        with open(stability_log_path, "a") as f:
                                            f.write(f"{self.state.global_step},{module_name},{module_type},{module_stability:.4f},{len(importances_list)}\n")
                                         
                                        print(f"[稳定性应用] module={module_name}, type={module_type}, stability={module_stability:.4f}, samples={len(importances_list)}")
                                         
                                        # 可选：保存每个头/神经元的稳定性详情
                                        detail_path = os.path.join(self.args.output_dir, f"stability_{module_type}_{module_name.replace('.', '_')}.csv")
                                        with open(detail_path, "a") as f:
                                            f.write(f"step,{self.state.global_step}," + ",".join([f"{s:.4f}" for s in stability]) + "\n")
                             
                            # 重置收集的重要性分数，准备下一个周期
                            self.collected_importances = {}
                             
                            # 如果计算了稳定性分数，添加到日志
                            if stability_scores:
                                attn_scores = {k: v for k, v in stability_scores.items() if v['type'] == 'attention'}
                                ffn_scores = {k: v for k, v in stability_scores.items() if v['type'] == 'ffn'}
                                print(f"[稳定性应用] 已成功计算 {len(attn_scores)} 个注意力模块和 {len(ffn_scores)} 个FFN模块的稳定性并更新敏感度字典")
                                 
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
                            import traceback
                            traceback.print_exc()  # 打印完整错误信息
                     
                    # 在剪枝周期开始时清空收集的数据
                    if self.state.global_step % self.prune_freq == 0:
                        self.collected_importances = {}
                        print(f"[稳定性采集] step={self.state.global_step}, 新剪枝周期开始，重置收集的重要性分数")
                    # ====== 稳定性逻辑结束 ======

                    # 更新敏感度字典
                    sensitivity_dict = self.update_sensitivity_with_profile(model, sensitivity_dict, self.prune_metric)
                    
                    # 计算当前稀疏率
                    ratio = utils.schedule_sparsity_ratio(self.state.global_step, self.state.max_steps,
                                                           self.warmup_iters, self.cooldown_iters, 
                                                           self.init_ratio, self.mac)
                    
                    # 执行剪枝（如果需要）
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
                            
                            # 打印日志
                            print(f"[性能监测] 步骤 {self.state.global_step} - 剪枝搜索: "
                                  f"{pruning_time:.2f}ms, 内存: {pruning_mem:.2f}MB, FLOPs: {pruning_flops:.2f}GF")
                    
                    # 应用掩码
                    utils.apply_masked_modules(model, self.masks)

                    # 执行优化器步骤
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                    
                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        # 添加剩余的tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        # 计算训练指标
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)

        # 训练结束回调
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # 生成完整的性能报告
        self.generate_full_training_report()

        return TrainOutput(self.state.global_step, train_loss, metrics) 

    def update_sensitivity_with_profile(self, model, sensitivity_dict, prune_metric):
        """监测更新敏感度字典的性能，并收集全面统计数据"""
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
        importance_flops = self.estimate_flops('importance')
        
        # 记录性能数据
        self.perf_stats['importance_time'].append((end_time - start_time))  # 以秒为单位存储
        
        # 记录当前方法的统计数据
        if prune_metric in self.method_stats:
            self.method_stats[prune_metric]['time'].append((end_time - start_time))
            self.method_stats[prune_metric]['memory'].append(memory_used)
        
        # 周期性记录日志
        if self.state.global_step % 50 == 0 or self.state.global_step in self.profile_steps:
            print(f"[性能监测] 步骤 {self.state.global_step} - 重要性计算({prune_metric}): "
                  f"{importance_time:.2f}ms, 内存: {memory_used:.2f}MB, FLOPs: {importance_flops:.2f}GF")
        
        return updated_dict

    def estimate_flops(self, phase):
        """估算不同阶段的FLOPs"""
        model = self.model
        batch_size = self.args.per_device_train_batch_size
        seq_length = self.seq_len
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
            method = self.prune_metric
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

    def generate_full_training_report(self):
        """生成全面的训练性能报告"""
        if not hasattr(self, 'perf_stats') or not self.perf_stats['forward_time']:
            return
        
        # 计算平均性能指标
        avg_forward_time = np.mean([t * 1000 for t in self.perf_stats['forward_time']])  # ms
        avg_backward_time = np.mean([t * 1000 for t in self.perf_stats['backward_time']])  # ms
        
        # 计算重要性方法的性能
        importance_times = {}
        importance_memory = {}
        for method, stats in self.method_stats.items():
            if stats['time']:
                importance_times[method] = np.mean([t * 1000 for t in stats['time']])  # ms
                importance_memory[method] = np.mean(stats['memory'])  # MB
        
        # 计算稳定性相关统计
        stability_stats = {}
        if hasattr(self, 'stability_stats') and self.stability_stats:
            # 计算每个模块类型的平均稳定性
            attention_stabilities = []
            ffn_stabilities = []
            
            for module, data in self.stability_stats.items():
                for entry in data:
                    avg_stability = np.mean(entry['stability'])
                    if 'attention' in module:
                        attention_stabilities.append(avg_stability)
                    elif 'intermediate' in module or 'output' in module:
                        ffn_stabilities.append(avg_stability)
            
            if attention_stabilities:
                stability_stats['attention'] = {
                    'mean': np.mean(attention_stabilities),
                    'min': np.min(attention_stabilities),
                    'max': np.max(attention_stabilities)
                }
            
            if ffn_stabilities:
                stability_stats['ffn'] = {
                    'mean': np.mean(ffn_stabilities),
                    'min': np.min(ffn_stabilities),
                    'max': np.max(ffn_stabilities)
                }
        
        # 创建报告文件
        report_path = os.path.join(self.args.output_dir, "final_training_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("最终训练性能报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本训练信息
            f.write("## 基本训练信息\n\n")
            f.write(f"- 模型: {self.model.__class__.__name__}\n")
            f.write(f"- 批大小: {self.args.per_device_train_batch_size}\n")
            f.write(f"- 序列长度: {self.seq_len}\n")
            f.write(f"- 总步数: {self.state.global_step}\n")
            f.write(f"- 稀疏率目标: {self.mac}\n")
            f.write(f"- 实际稀疏率: {self.mac_ratio:.4f}\n")
            f.write(f"- 稳定性权重: {self.stability_weight}\n")
            f.write(f"- 稳定性组件: {self.stability_components}\n\n")
            
            # 性能统计
            f.write("## 性能统计\n\n")
            f.write(f"- 平均前向时间: {avg_forward_time:.2f}ms\n")
            f.write(f"- 平均反向时间: {avg_backward_time:.2f}ms\n")
            
            for method, time in importance_times.items():
                f.write(f"- 平均{method}重要性计算时间: {time:.2f}ms\n")
                f.write(f"- 平均{method}重要性计算内存: {importance_memory[method]:.2f}MB\n")
            
            # 稳定性统计
            if stability_stats:
                f.write("\n## 稳定性统计\n\n")
                
                if 'attention' in stability_stats:
                    attn = stability_stats['attention']
                    f.write(f"- 注意力头稳定性: 平均={attn['mean']:.4f}, 最小={attn['min']:.4f}, 最大={attn['max']:.4f}\n")
                
                if 'ffn' in stability_stats:
                    ffn = stability_stats['ffn']
                    f.write(f"- FFN层稳定性: 平均={ffn['mean']:.4f}, 最小={ffn['min']:.4f}, 最大={ffn['max']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"已生成最终训练报告: {report_path}")

    def save_stability_distribution(self, output_path=None):
        """保存稳定性分数分布数据用于后续分析"""
        if not hasattr(self, 'stability_stats') or not self.stability_stats:
            print("未找到稳定性统计数据")
            return
        
        if output_path is None:
            output_path = os.path.join(self.args.output_dir, "stability_distribution.npz")
        
        # 将稳定性数据转换为易于分析的格式
        analysis_data = {
            'attention': {},
            'ffn': {}
        }
        
        for module_name, data_list in self.stability_stats.items():
            module_type = "attention" if "attention" in module_name else "ffn"
            
            # 初始化该模块的数据
            if module_name not in analysis_data[module_type]:
                analysis_data[module_type][module_name] = {
                    'steps': [],
                    'stability_values': [],
                    'num_samples': []
                }
            
            # 添加数据
            for entry in data_list:
                analysis_data[module_type][module_name]['steps'].append(entry['step'])
                analysis_data[module_type][module_name]['stability_values'].append(entry['stability'])
                analysis_data[module_type][module_name]['num_samples'].append(entry['num_samples'])
        
        # 保存为numpy压缩文件
        np.savez_compressed(output_path, data=analysis_data)
        print(f"稳定性分布数据已保存至: {output_path}")
        
        # 同时保存一个更易于人类阅读的摘要
        summary_path = os.path.join(os.path.dirname(output_path), "stability_distribution_summary.txt")
        with open(summary_path, "w") as f:
            f.write("稳定性分数分布摘要\n")
            f.write("=" * 60 + "\n\n")
            
            attn_modules = list(analysis_data['attention'].keys())
            ffn_modules = list(analysis_data['ffn'].keys())
            
            f.write(f"共分析了 {len(attn_modules)} 个注意力模块和 {len(ffn_modules)} 个FFN模块\n\n")
            
            if attn_modules:
                f.write("## 注意力模块\n\n")
                for module in attn_modules:
                    data = analysis_data['attention'][module]
                    last_stability = data['stability_values'][-1] if data['stability_values'] else []
                    if len(last_stability) > 0:
                        mean_stability = np.mean(last_stability)
                        min_stability = np.min(last_stability)
                        max_stability = np.max(last_stability)
                        std_stability = np.std(last_stability)
                        
                        f.write(f"模块: {module}\n")
                        f.write(f"  样本数: {data['num_samples'][-1] if data['num_samples'] else 0}\n")
                        f.write(f"  平均稳定性: {mean_stability:.4f}\n")
                        f.write(f"  最小稳定性: {min_stability:.4f}\n")
                        f.write(f"  最大稳定性: {max_stability:.4f}\n")
                        f.write(f"  标准差: {std_stability:.4f}\n\n")
            
            if ffn_modules:
                f.write("## FFN模块\n\n")
                for module in ffn_modules:
                    data = analysis_data['ffn'][module]
                    last_stability = data['stability_values'][-1] if data['stability_values'] else []
                    if len(last_stability) > 0:
                        mean_stability = np.mean(last_stability)
                        min_stability = np.min(last_stability)
                        max_stability = np.max(last_stability)
                        std_stability = np.std(last_stability)
                        
                        f.write(f"模块: {module}\n")
                        f.write(f"  样本数: {data['num_samples'][-1] if data['num_samples'] else 0}\n")
                        f.write(f"  平均稳定性: {mean_stability:.4f}\n")
                        f.write(f"  最小稳定性: {min_stability:.4f}\n")
                        f.write(f"  最大稳定性: {max_stability:.4f}\n")
                        f.write(f"  标准差: {std_stability:.4f}\n\n")
        
        print(f"稳定性分布摘要已保存至: {summary_path}")
        return output_path

    def generate_stability_report(self, report_path=None):
        """生成详细的稳定性报告，包括稳定性与性能关联分析"""
        if not hasattr(self, 'stability_stats') or not self.stability_stats:
            print("未找到稳定性统计数据")
            return
        
        if report_path is None:
            report_path = os.path.join(self.args.output_dir, "detailed_stability_report.md")
        
        # 收集注意力和FFN的稳定性数据
        attention_data = []
        ffn_data = []
        
        for module_name, data_list in self.stability_stats.items():
            is_attention = 'attention' in module_name
            for entry in data_list:
                item = {
                    'module': module_name,
                    'step': entry['step'],
                    'stability': entry['stability'],
                    'num_samples': entry['num_samples']
                }
                if is_attention:
                    attention_data.append(item)
                else:
                    ffn_data.append(item)
        
        # 根据步骤对数据进行排序
        attention_data.sort(key=lambda x: x['step'])
        ffn_data.sort(key=lambda x: x['step'])
        
        # 写入报告
        with open(report_path, "w") as f:
            f.write("# 稳定性分析详细报告\n\n")
            
            f.write("## 实验配置\n\n")
            f.write(f"- 使用稳定性: {self.use_stability}\n")
            f.write(f"- 稳定性组件: {self.stability_components}\n")
            f.write(f"- 稳定性权重: {self.stability_weight}\n")
            f.write(f"- 稳定性收集比例: {self.stability_collection_ratio}\n\n")
            
            f.write("## 注意力模块稳定性\n\n")
            if attention_data:
                # 按模块分组
                modules = {}
                for item in attention_data:
                    module = item['module']
                    if module not in modules:
                        modules[module] = []
                    modules[module].append(item)
                
                f.write(f"共分析了 {len(modules)} 个注意力模块\n\n")
                
                for module, items in modules.items():
                    # 考虑最后一个条目，这是最新的数据
                    latest = items[-1]
                    stabilities = latest['stability']
                    
                    # 计算统计信息
                    mean_stability = np.mean(stabilities)
                    min_stability = np.min(stabilities)
                    max_stability = np.max(stabilities)
                    std_stability = np.std(stabilities)
                    
                    f.write(f"### 模块: {module}\n\n")
                    f.write(f"- 最后一次收集步骤: {latest['step']}\n")
                    f.write(f"- 样本数: {latest['num_samples']}\n")
                    f.write(f"- 平均稳定性: {mean_stability:.4f}\n")
                    f.write(f"- 最小稳定性: {min_stability:.4f}\n")
                    f.write(f"- 最大稳定性: {max_stability:.4f}\n")
                    f.write(f"- 标准差: {std_stability:.4f}\n\n")
                    
                    # 添加稳定性分布图表描述
                    f.write("#### 稳定性分布\n\n")
                    f.write("稳定性分数从低到高排序，低稳定性表示头的排名变化较大。\n\n")
                    
                    # 添加最稳定和最不稳定的头的分析
                    sorted_indices = np.argsort(stabilities)
                    most_stable = sorted_indices[:3]
                    least_stable = sorted_indices[-3:]
                    
                    f.write("#### 稳定性排名\n\n")
                    f.write("最稳定的头 (排名变化最小):\n\n")
                    for i, idx in enumerate(most_stable):
                        f.write(f"- 头 #{idx}: 稳定性分数 = {stabilities[idx]:.4f}\n")
                    
                    f.write("\n最不稳定的头 (排名变化最大):\n\n")
                    for i, idx in enumerate(least_stable):
                        f.write(f"- 头 #{idx}: 稳定性分数 = {stabilities[idx]:.4f}\n")
                    
                    f.write("\n---\n\n")
            else:
                f.write("没有收集到注意力模块的稳定性数据\n\n")
            
            f.write("## FFN模块稳定性\n\n")
            if ffn_data:
                # 按模块分组
                modules = {}
                for item in ffn_data:
                    module = item['module']
                    if module not in modules:
                        modules[module] = []
                    modules[module].append(item)
                
                f.write(f"共分析了 {len(modules)} 个FFN模块\n\n")
                
                for module, items in modules.items():
                    # 考虑最后一个条目，这是最新的数据
                    latest = items[-1]
                    stabilities = latest['stability']
                    
                    # 计算统计信息
                    mean_stability = np.mean(stabilities)
                    min_stability = np.min(stabilities)
                    max_stability = np.max(stabilities)
                    std_stability = np.std(stabilities)
                    
                    f.write(f"### 模块: {module}\n\n")
                    f.write(f"- 最后一次收集步骤: {latest['step']}\n")
                    f.write(f"- 样本数: {latest['num_samples']}\n")
                    f.write(f"- 平均稳定性: {mean_stability:.4f}\n")
                    f.write(f"- 最小稳定性: {min_stability:.4f}\n")
                    f.write(f"- 最大稳定性: {max_stability:.4f}\n")
                    f.write(f"- 标准差: {std_stability:.4f}\n\n")
                    
                    # 添加稳定性分布图表描述
                    f.write("#### 稳定性分布\n\n")
                    f.write("稳定性分数从低到高排序，低稳定性表示神经元的排名变化较大。\n\n")
                    
                    # 添加最稳定和最不稳定的神经元的分析
                    sorted_indices = np.argsort(stabilities)
                    most_stable = sorted_indices[:3]
                    least_stable = sorted_indices[-3:]
                    
                    f.write("#### 稳定性排名\n\n")
                    f.write("最稳定的神经元 (排名变化最小):\n\n")
                    for i, idx in enumerate(most_stable):
                        f.write(f"- 神经元 #{idx}: 稳定性分数 = {stabilities[idx]:.4f}\n")
                    
                    f.write("\n最不稳定的神经元 (排名变化最大):\n\n")
                    for i, idx in enumerate(least_stable):
                        f.write(f"- 神经元 #{idx}: 稳定性分数 = {stabilities[idx]:.4f}\n")
                    
                    f.write("\n---\n\n")
            else:
                f.write("没有收集到FFN模块的稳定性数据\n\n")
            
            f.write("## 稳定性与性能相关分析\n\n")
            f.write("本节分析稳定性对模型性能的潜在影响。\n\n")
            
            f.write("### 消融实验建议\n\n")
            f.write("为了验证稳定性分数的有效性，建议进行以下消融实验：\n\n")
            f.write("1. **基线模型**：不使用稳定性分数进行剪枝\n")
            f.write("2. **注意力稳定性**：仅对注意力头使用稳定性分数\n")
            f.write("3. **FFN稳定性**：仅对FFN层使用稳定性分数\n")
            f.write("4. **全模型稳定性**：对注意力头和FFN层同时使用稳定性分数\n")
            f.write("5. **稳定性权重变化**：尝试不同的稳定性权重，如0.1, 0.3, 0.5, 0.7\n\n")
            
            f.write("使用以下指标比较这些实验：\n\n")
            f.write("- 任务性能（准确率、F1分数等）\n")
            f.write("- 训练收敛速度\n")
            f.write("- 剪枝后模型的稳定性\n")
            f.write("- 推理速度变化\n\n")
        
        print(f"详细稳定性报告已生成：{report_path}")
        return report_path 