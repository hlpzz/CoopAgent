from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOConfig
import torch
from datasets import load_dataset

# 导入自定义trainer
from .padpo_trainer import CustomDPOTrainer, CustomDPODataCollator

# Replace these placeholders before training. MODEL_PATH may also be a Hugging Face model ID.
MODEL_PATH = "path/to/model"
ADAPTER_PATH = "path/to/sft_adapter"
TRAIN_DATA_PATH = "path/to/preference_dataset.json"
OUTPUT_DIR = "path/to/padpo_output"


# ====== 加载 tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
# ====== 加载基础模型并应用LoRA adapter ======
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

# ====== 加载LoRA adapter ======
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    torch_dtype=torch.bfloat16,  # 明确指定dtype
    is_trainable=True,
)
model = model.to(dtype=torch.bfloat16)



# ====== 加载 DPO 数据集 ======
dataset = load_dataset("json", data_files=TRAIN_DATA_PATH)
train_dataset = dataset["train"]

# ====== LoRA 配置 ======
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    use_dora=False
)

# ====== DPO 训练参数 ======
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=1,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    ddp_find_unused_parameters=False,

    max_prompt_length=8*1024,
    max_length=8*1024,
    beta=0.1,
    loss_type="sigmoid",
    label_pad_token_id=-100,
    padding_value=0,
    generate_during_eval=False,
    disable_dropout=True,

    remove_unused_columns=False,
)

# ====== 创建自定义data collator ======
data_collator = CustomDPODataCollator(
    pad_token_id=tokenizer.pad_token_id,
    label_pad_token_id=training_args.label_pad_token_id,
    is_encoder_decoder=False,
)

# ====== 初始化自定义DPOTrainer ======
trainer = CustomDPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    processing_class=tokenizer,
    data_collator=data_collator,
    regularization_weight=0.001,
)

# ====== 开始训练 ======
trainer.train()
