from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOConfig
import torch
from datasets import load_dataset

# 导入自定义trainer
from .padpo_trainer import CustomDPOTrainer, CustomDPODataCollator

# ====== 路径配置 ======
base_model_path = "/home/zhengzhe/qwen2.5-7B-Instruct"
adapter_path = "/home/zhengzhe/SFT/dedup_lora_result/checkpoint-1575"
# dpo_data_path = "/home/zhengzhe/dataset/temp_dataset.json"
# dpo_data_path = "/home/zhengzhe/dataset/dpo_noise_clean_dataset.json"
dpo_data_path = "/home/zhengzhe/dataset/dpo_noise_short_dataset.json"


# ====== 加载 tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
# ====== 加载基础模型并应用LoRA adapter ======
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,  # ← 改为auto，让transformers自动管理
)

# ====== 加载LoRA adapter ======
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    torch_dtype=torch.bfloat16,  # 明确指定dtype
    is_trainable=True,
)
model = model.to(dtype=torch.bfloat16)

# ref_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.bfloat16,
#     device_map=None,
#     trust_remote_code=True,
# )
# for param in ref_model.parameters():
#     param.requires_grad = False
# ref_model = PeftModel.from_pretrained(
#     ref_base,
#     adapter_path,
#     is_trainable=False,  # Frozen
# )


# ====== 加载 DPO 数据集 ======
dataset = load_dataset("json", data_files=dpo_data_path)
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
    output_dir="/home/zhengzhe/DPO/dpo_relu_noise_result",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_dir="/home/zhengzhe/DPO/logs",
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
