from transformers.trainer import *
import loraprune.utils as utils
import loraprune.utils1 as utils1
from peft.tuners.lora import Linear
from torch.utils.data.distributed import DistributedSampler
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
                 seq_len
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
        # self.memory_log = []

    # def log_memory_usage(self, stage):
    #     allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    #     reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    #     log_msg = f"[{stage}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
    #     # print(log_msg)
    #     self.memory_log.append((stage, allocated, reserved))
    #     with open("memory_log.txt", "a") as log_file:
    #         log_file.write(log_msg + "\n")
    # def training_step(self, model, inputs):
    #     model.train()
    #     self.log_memory_usage("Before Forward")
    #     outputs = model(**inputs)
    #     self.log_memory_usage("After Forward")
    #     loss = outputs.loss
    #     loss.backward()
    #     self.log_memory_usage("After Backward")
    #     self.optimizer.step()
    #     self.log_memory_usage("After Optimizer Step")
    #     self.optimizer.zero_grad()
    #     return loss        

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
                                if 'attention' in layer_name:
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
                                        
                                elif 'output' in layer_name or 'intermediate' in layer_name:
                                    # 对于输出层和中间层，调整掩码形状
                                    mask = mask.view(1, 1, -1)  # 变成 [1, 1, 3072]
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
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
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
                if self.deepspeed:
                    self.deepspeed.step()

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    if not self.deepspeed:
                        # print("prune_metric", self.prune_metric)
                        sensitivity_dict = utils.update_sensitivity_dict(model, sensitivity_dict, self.prune_metric)
                        # print("sensitivity_dict", sensitivity_dict)
                    ratio = utils.schedule_sparsity_ratio(self.state.global_step, self.state.max_steps,
                                                          self.warmup_iters,
                                                          self.cooldown_iters, self.init_ratio, self.mac)
                    # print("ratio", ratio)                
                    # ratio = 0.05
                    if (self.state.global_step) % self.prune_freq == 0 and ratio > self.init_ratio:
                    # if (self.state.global_step) % self.prune_freq == 0 and self.masks is None:
                        print("prunning at steps", self.state.global_step)
                        if self.mac_ratio < self.mac:
                            print("Pruning condition met, calling local_prune...")
                        # self.masks = utils.local_prune_change(model, sensitivity_dict, ratio, self.ratio, self.masks)
                            # new_masks= {}
                        # new_masks = utils.local_prune(model, sensitivity_dict, ratio, self.ratio, new_masks)
                        #一次性剪枝
                            # new_masks ,mac_ratio= utils.search_mac_change(model, sensitivity_dict,self.seq_len,ratio,new_masks)
                            self.masks,mac_ratio = utils.search_mac_change(model, sensitivity_dict,self.seq_len,ratio,self.masks)
                            self.mac_ratio = mac_ratio
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
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                    # if self.masks is not None:
                        # print("Applying mask after optimizer step")
                        # utils1.apply_model_mask(model, self.masks)
                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
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
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


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
        # # 生成完整的性能报告
        # _ = self.generate_full_training_report()

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
