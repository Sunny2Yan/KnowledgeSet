from typing import Optional, Literal


class PPOConfig:
    # common parameters
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    task_name: Optional[str] = None
    model_name: Optional[str] = "gpt2"
    query_dataset: Optional[str] = "imdb"
    reward_model: Optional[str] = "sentiment-analysis:lvwerra/distilbert-imdb"
    """The reward model to use - used only for tracking purposes"""
    remove_unused_columns: bool = True
    """Remove unused columns from the dataset if `datasets.Dataset` is used"""
    tracker_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the tracker (e.g. python ppo.py --tracker_kwargs='{"wandb": {"entity": "my_wandb_entity", "name": "my_exp_name"}}'"""
    accelerator_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator"""
    project_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator project config (e.g. `logging_dir`)"""
    tracker_project_name: str = "trl"
    """Name of project to use for tracking"""
    push_to_hub_if_best_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for pushing model to the hub during training (e.g. repo_id)"""

    # hyperparameters
    steps: int = 20000
    """Number of training steps"""
    learning_rate: float = 1.41e-5
    """Adam learning rate"""
    adap_kl_ctrl: bool = True
    """Use adaptive KL control, otherwise linear"""
    init_kl_coef: Optional[float] = 0.2
    """Initial KL penalty coefficient (used for adaptive and linear control)"""
    kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl"
    """kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"""
    target: Optional[float] = 6
    """Target KL value for adaptive KL control"""
    horizon: Optional[float] = 10000
    """Horizon for adaptive KL control"""
    gamma: float = 1
    """Gamma parameter for advantage calculation"""
    lam: float = 0.95
    """Lambda parameter for advantage calculation"""
    cliprange: float = 0.2
    """Range for clipping in PPO policy gradient loss"""
    cliprange_value: float = 0.2
    """Range for clipping values in loss calculation"""
    vf_coef: float = 0.1
    """Scaling factor for value loss"""
    batch_size: int = 128
    """Number of samples per optimisation step"""
    forward_batch_size: Optional[int] = None
    """DEPRECATED: use `mini_batch_size` instead, which does the same thing."""
    mini_batch_size: int = 128
    """Number of samples optimized in each mini batch"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    world_size: tyro.conf.Suppress[int] = None
    """The world size for distributed training"""
    ppo_epochs: int = 4
    """Number of optimisation epochs per batch of samples"""
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping"""
    optimize_cuda_cache: Optional[bool] = None
    """DEPRECATED: use `optimize_device_cache` instead, which does the same thing."""
    optimize_device_cache: Optional[bool] = False
    """Optimize device cache for slightly more memory-efficient training"""
    early_stopping: bool = False
    """Whether to stop the PPO optimization loop early is the KL too high"""
    target_kl: float = 1
    """Stop early if we exceed this value by over 50%"""
    compare_steps: int = 1
    """Number of steps between comparison of the current reward with the best seen so far"""
    ratio_threshold: float = 10.0
    """Skip mini-batches with high PPO ratios that can cause loss spikes"""
    use_score_scaling: bool = False
    """Use score scaling"""
    use_score_norm: bool = False
    """Use score normalization. Only applicable if use_score_scaling is True"""
    score_clip: Optional[float] = None
    """Score clipping"""
    whiten_rewards: bool = False
    """Whiten the rewards before compute advantages"""

    # computed hyperparameters at runtime; we use `tyro.conf.Suppress` to hide them from the help text
    is_encoder_decoder: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is an encoder-decoder model"""
    is_peft_model: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is a PEFT model"""
    backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: Number of samples optimized in an `optimizer.step()` call"""
    global_backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `backward_batch_size` across all processes"""
    global_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `batch_size` across all processes"""

    if optimize_cuda_cache is not None:
        warnings.warn(
            "The `optimize_cuda_cache` argument will be deprecated soon, please use `optimize_device_cache` instead."
        )
        optimize_device_cache = optimize_cuda_cache
    else:
        optimize_device_cache = False

    def __post_init__(self):
        if self.forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for PPO optimization."
            )
            self.mini_batch_size = self.forward_batch_size

        self.backward_batch_size = self.mini_batch_size * self.gradient_accumulation_steps
        exact_div(
            self.batch_size,
            self.backward_batch_size,
            "`batch_size`",
            "`mini_batch_size * gradient_accumulation_steps`",
            "`batch_size` must be a multiple of `mini_batch_size * gradient_accumulation_steps`",
        )

        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            if not is_wandb_available():
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]


from torch.utils.data import DataLoader


class PPOTrainer:
    def __init__(
        self,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        # Step 1.1 Runtime variables filled by the accelerator
        config.world_size = self.accelerator.num_processes
        config.global_backward_batch_size = config.backward_batch_size * config.world_size
        config.global_batch_size = config.batch_size * config.world_size

        self.tokenizer = tokenizer
        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {SUPPORTED_ARCHITECTURES} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            )
        lr_scheduler_class = (torch.optim.lr_scheduler._LRScheduler
                              if not is_torch_greater_2_0()
                              else torch.optim.lr_scheduler.LRScheduler)

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.num_processes > 1

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            if is_xpu_available():
                self.current_device = torch.device("xpu:0")
            elif is_npu_available():
                self.current_device = torch.device("npu:0")
            else:
                self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_device_cache = self.config.optimize_device_cache

        self.running = RunningMoments(self.accelerator)



    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def loss(self, *args):
        raise NotImplementedError("Not implemented")

    def compute_rewards(self, *args):
        raise NotImplementedError("Not implemented")

    def _save_pretrained(self, save_directory):
        raise NotImplementedError("Not implemented")