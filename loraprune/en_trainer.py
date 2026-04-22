from transformers.trainer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os
import json
from peft.tuners.lora import Linear

# 导入LoRAPruneTrainer
from loraprune.trainer import LoRAPruneTrainer
import loraprune.utils as utils

class EnhancedLoRAPruneTrainer(LoRAPruneTrainer):
    def __init__(
        self, 
        model,
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
        teacher_model=None,
        sparsity_lambda=1e-4,
        distill_lambda=0.5,
        structure_lambda=0.1,
        temperature=2.0,
        distill_warmup_steps=1000,
    ):
        # 调用父类LoRAPruneTrainer的初始化
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            args=args,
            data_collator=data_collator,
            ratio=ratio,
            init_ratio=init_ratio,
            warmup_iters=warmup_iters,
            cooldown_iters=cooldown_iters,
            prune_freq=prune_freq,
            prune_metric=prune_metric,
            mac=mac,
            seq_len=seq_len
        )
        
        # 添加额外的知识蒸馏相关参数
        self.teacher_model = teacher_model
        self.sparsity_lambda = sparsity_lambda
        self.base_distill_lambda = distill_lambda
        self.structure_lambda = structure_lambda
        self.temperature = temperature
        self.distill_warmup_steps = distill_warmup_steps
        
        # 用于记录损失
        self.loss_log = {
            'task_loss': [],
            'distill_loss': [],
            'sparsity_loss': [],
            'structure_loss': [],
            'total_loss': []
        }

    def get_distill_lambda(self):
        """动态调整蒸馏权重"""
        if self.state.global_step < self.distill_warmup_steps:
            return self.base_distill_lambda * (self.state.global_step / self.distill_warmup_steps)
        return self.base_distill_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        """重写计算损失函数以添加蒸馏损失"""
        # 1. 计算原始任务损失
        outputs = model(**inputs)
        task_loss = outputs.loss
        
        total_loss = task_loss
        loss_dict = {'task_loss': task_loss.item()}

        # 2. 知识蒸馏损失
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                teacher_conf = torch.max(teacher_probs, dim=-1)[0].mean()

            current_distill_lambda = self.get_distill_lambda() * teacher_conf.item()
            
            distill_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(outputs.logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1)
            ) * (self.temperature ** 2)
            
            total_loss += current_distill_lambda * distill_loss
            loss_dict['distill_loss'] = distill_loss.item()
            loss_dict['teacher_confidence'] = teacher_conf.item()

        # 3. 稀疏正则化损失
        sparsity_loss = 0
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                sparsity_loss += torch.norm(module.weight, p=1)
        total_loss += self.sparsity_lambda * sparsity_loss
        loss_dict['sparsity_loss'] = sparsity_loss.item()

        # 4. 结构相似性损失
        structure_loss = 0
        if hasattr(self, 'masks') and self.masks:
            for name, module in model.named_modules():
                if isinstance(module, Linear) and name in self.masks:
                    weight_structure = torch.abs(module.weight) > 0
                    mask_structure = self.masks[name].bool()
                    structure_loss += F.binary_cross_entropy(
                        weight_structure.float(),
                        mask_structure.float()
                    )
            total_loss += self.structure_lambda * structure_loss
            loss_dict['structure_loss'] = structure_loss.item()

        # 记录损失
        for key, value in loss_dict.items():
            if key in self.loss_log:
                self.loss_log[key].append(value)
        self.loss_log['total_loss'].append(total_loss.item())

        # 每100步打印一次损失统计
        if self.state.global_step % 100 == 0:
            self._log_loss_statistics()

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _log_loss_statistics(self):
        """打印损失统计信息"""
        print(f"\nStep {self.state.global_step} Loss Statistics:")
        for key in self.loss_log:
            if self.loss_log[key]:
                recent_losses = self.loss_log[key][-100:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"{key}: {avg_loss:.4f}")

    def train(self):
        """重写train方法以添加教师模型评估"""
        print("开始训练...")
        if self.teacher_model is not None:
            print("评估教师模型性能...")
            teacher_trainer = transformers.Trainer(
                model=self.teacher_model,
                args=self.args,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics
            )
            teacher_metrics = teacher_trainer.predict(self.eval_dataset)
            print(f"教师模型性能: {teacher_metrics}")

        return super().train()
