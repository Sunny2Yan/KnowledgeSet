from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class PPOConfig:
    # common parameters
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    task_name: Optional[str] = None
    model_name: Optional[str] = "gpt2"
    query_dataset: Optional[str] = "imdb"
    reward_model: Optional[str] = "sentiment-analysis:lvwerra/distilbert-imdb"
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

        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]


import torch
import random
import numpy as np
from torch.utils.data import DataLoader


class PPOTrainer:
    """
    Attributes:
        **config** (`PPOConfig`) -- PPOTrainer 的配置对象。查看 `PPOConfig` 的文档以了解更多细节。
        **model** (`PreTrainedModelWrapper`) -- 要优化的模型，带有值头的 Hugging Face 变换器模型。
            查看 `PreTrainedModelWrapper` 的文档以了解更多细节。
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- 用于 KL 惩罚的参考模型，带有随意语言建模头的 Hugging Face
            变换器模型。查看 `PreTrainedModelWrapper` 的文档以了解更多细节。如果未提供参考模型，训练器将使用与要优化的模型相同体系结构的参考模型
            创建共享层。
        **tokenizer** (`PreTrainedTokenizerBase`) -- 用于编码数据的分词器。查看 `transformers.PreTrainedTokenizer` 和
            `transformers.PreTrainedTokenizerFast` 的文档以了解更多细节。
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch 数据集或 Hugging
            Face 数据集。用于创建 PyTorch 数据加载器。如果未提供数据集，则必须在训练器外部创建数据加载器，用户需要设计自己的数据加载器并确保批量
            大小与配置对象中指定的相同。
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- 用于训练的优化器。如果未提供优化器，训练器将使用配置对象中指定的学习率创建 Adam 优化器。
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- 用于训练和传递给数据加载器的数据收集器。
        **num_shared_layers** (int, *optional*) -- 模型和参考模型之间要共享的层数，如果未传递参考模型。如果未提供数量，则将共享所有层。
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- 用于训练的学习率调度器。
    """
    def __init__(
        self, config, tokenizer, model, dataset, data_collator,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        num_shared_layers: Optional[int] = None,
    ):
        """
        初始化 PPOTrainer。

        Args:
            config (`PPOConfig`):
                PPOTrainer 的配置对象。查看 `PPOConfig` 的文档以了解更多细节。
            model (`PreTrainedModelWrapper`):
                带有值头的 Hugging Face 变换器模型。
            ref_model (`PreTrainedModelWrapper`):
                带有随意语言建模头的 Hugging Face 变换器模型。用于 KL 惩罚。
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face 分词器。
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch 数据集或 Hugging Face 数据集。如果传递了 Hugging Face 数据集，将通过删除模型不使用的列来预处理数据集。
                如果未传递数据集，则会在多 GPU 设置中引发警告。
            optimizer (Optional[`torch.optim.Optimizer`]):
                用于训练的优化器。如果为 `None`，则默认使用 `Adam`。
            data_collator (Optional[function]):
                数据收集器函数。
            num_shared_layers (Optional[int]):
                模型和参考模型之间要共享的层数。如果为 `None`，则共享所有层。
                仅当未传递 `ref_model` 时使用。
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                用于训练的学习率调度器。
        """
        self.config = config
        self.set_seed(config.seed)

        # step 0：检查参数的有效性
        # step 1：初始化加速器
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )
        # 步骤 1.1 加速器填充的运行时变量
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
                    "当提供 ref_model 时，将忽略 num_shared_layers。两个不同的模型用于模型和参考模型，且不共享任何层。",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model 必须是 PreTrainedModelWrapper 或 `None`，得到 {type(ref_model)} - 支持的架构有：{SUPPORTED_ARCHITECTURES}"
            )

        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model else nullcontext
        )

        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "未提供数据集。在多 GPU 设置中，这将导致错误。您应该自行准备数据加载器，使用 `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
                " 并使用 `torch.utils.data.DataLoader`，或将数据集传递给 `PPOTrainer`。请参阅文档了解更多详细信息。",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # 步骤 3：初始化优化器和数据收集器
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.config.learning_rate, )
        self.lr_scheduler = torch.optim.lr_scheduler.LRScheduler(self.optimizer)


        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # DeepSpeed 集成的安全检查器
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
            # 量化模型已设置在正确的设备上
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # 在分布式设置中，只需在主进程上执行日志记录
        self.is_distributed = self.accelerator.num_processes > 1

        # 初始化当前步骤
        self.current_step = 0

        # 用于 PP 的后处理
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_device_cache = self.config.optimize_device_cache

        self.running = RunningMoments(self.accelerator)

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':
    max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number.

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/zephyr-sft",
        max_seq_length = max_seq_length,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 is currently optimized
        bias = "none",    # Bias = "none" is currently optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    training_args = TrainingArguments(output_dir="./output")

    dpo_trainer = DPOTrainer(
        model,
        model_ref=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    ```


    ### Downsides to merging QLoRA before DPO (approach 2)
    ```python
    # Load the base model.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/mixtral-8x7b-v0.1",
        load_in_4bit=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # Load the adapter.
    model = PeftModel.from_pretrained(
        model,
        "/path/to/peft",
        is_trainable=True,
        adapter_name="train",
    )
    # Load the adapter a second time, with a different name, which will be our reference model.
    model.load_adapter("/path/to/peft", adapter_name="reference")

    # Initialize the trainer, without a ref_model param.
    dpo_trainer = DPOTrainer(
        model,
        ...
        model_adapter_name="train",
        ref_adapter_name="reference",
    )
    ```