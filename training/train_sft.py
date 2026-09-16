from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
import json
from datasets import Dataset
import bitsandbytes as bnb


# Replace these placeholders before training. MODEL_PATH may also be a Hugging Face model ID.
MODEL_PATH = "path/to/model"
TRAIN_DATA_PATH = "path/to/sft_dataset.json"
OUTPUT_DIR = "path/to/sft_output"


# ======== 加载模型  ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code = True)

tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = prepare_model_for_kbit_training(model)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ======== LoRA 配置 ========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                # LoRA 矩阵秩
    lora_alpha=16,       # LoRA scaling factor
    lora_dropout=0.05,   # Dropout 防止过拟合
    target_modules=["q_proj", "v_proj"],  # 只训练注意力层的 Q/V 权重
    use_dora = False
)

# 将 LoRA 适配到模型
model = get_peft_model(model, lora_config)

# ======== 读取 JSON 数据 ========

with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def preprocess_function(examples):
    input_text = examples['input']
    target_text = examples['output']

    # 仅进行截断，不进行填充
    input_ids = tokenizer(input_text, truncation=True, padding=False)
    target_ids = tokenizer(target_text, truncation=True, padding=False)

    # 获取实际的 token ID 序列
    input_ids_list = input_ids["input_ids"]
    target_ids_list = target_ids["input_ids"]
    target_ids_list.append(tokenizer.eos_token_id)

    # 统一填充到 max_length（保证 batch 训练对齐）
    max_length = 12 * 1024
    # 限制 input + target 总长度不超过 max_length
    max_input_length = max_length - len(target_ids_list)  # 预留 output 空间
    if len(input_ids_list) > max_input_length:
        input_ids_list = input_ids_list[:max_input_length]  # 截断 input

    # 合并 input 和 target
    final_input_ids = input_ids_list + target_ids_list  # input 和 target 合并

    # 计算 padding 长度
    padding_length = max_length - len(final_input_ids)
    final_input_ids += [tokenizer.pad_token_id] * padding_length

    labels = [-100] * len(input_ids_list)  # 默认 label 为 -100
    labels += target_ids_list
    labels += [-100] * padding_length

    # attention_mask：input + target 部分为 1，padding 部分为 0
    attention_mask = [1] * (len(input_ids_list) + len(target_ids_list)) + [0] * padding_length

    return {
        "input_ids": final_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_datasets = dataset.map(preprocess_function, batched=False)

# ======== 训练参数配置 ========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    save_total_limit=2,
    save_steps=8000,

    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",

    fp16=False,
    bf16=True,
    logging_steps=2,
    optim="adamw_torch_fused",
    label_names=["labels"],

    ddp_find_unused_parameters=False,   # LoRA时建议关闭
    gradient_checkpointing=True,
    dataloader_drop_last=True           # 保证每卡样本数一致
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    processing_class=tokenizer,
    optimizers=(None, None)
)

# ======== 训练 ========
trainer.train()
