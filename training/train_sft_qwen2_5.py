# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# from peft import get_peft_model, LoraConfig, TaskType
# import torch
# import json
# from datasets import Dataset
# import bitsandbytes as bnb

# # ======== 加载 Qwen2.5-7B-Instruct（4-bit 量化） ========
# model_name = "/home/zhengzhe/qwen2.5-7B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

# tokenizer.padding_side = "right"
# tokenizer.pad_token_id = tokenizer.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,  # 使用 4-bit 量化加载模型
#     device_map="auto"   # 自动分配到 GPU
# )

# # ======== LoRA 配置 ========
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=16,                # LoRA 矩阵秩
#     lora_alpha=32,       # LoRA scaling factor
#     lora_dropout=0.05,   # Dropout 防止过拟合
#     target_modules=["q_proj", "v_proj"]  # 只训练注意力层的 Q/V 权重
# )

# # 将 LoRA 适配到模型
# model = get_peft_model(model, lora_config)

# # ======== 读取 JSONL 数据 ========

# with open('/home/zhengzhe/TM_dataset.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# dataset = Dataset.from_list(data)

# import random

# def preprocess_function(examples):
#     input_text = examples['input']
#     target_text = examples['output']

#     # 仅进行截断，不进行填充
#     input_ids = tokenizer(input_text, truncation=True, padding=False)
#     target_ids = tokenizer(target_text, truncation=True, padding=False)

#     # 获取实际的 token ID 序列
#     input_ids_list = input_ids["input_ids"]
#     target_ids_list = target_ids["input_ids"]
#     target_ids_list.append(tokenizer.eos_token_id)
#     # print("="*80)
#     # print(f"input  len:{len(input_ids_list)}")
#     # print(f"output len:{len(target_ids_list)}")

#     # 统一填充到 max_length（保证 batch 训练对齐）
#     max_length = 8192
#     # 限制 input + target 总长度不超过 max_length
#     max_input_length = max_length - len(target_ids_list)  # 预留 output 空间
#     if len(input_ids_list) > max_input_length:
#         input_ids_list = input_ids_list[:max_input_length]  # 截断 input
    
#     # 合并 input 和 target
#     final_input_ids = input_ids_list + target_ids_list  # input 和 target 合并

#     # 计算 padding 长度
#     padding_length = max_length - len(final_input_ids)
#     final_input_ids += [tokenizer.pad_token_id] * padding_length

#     labels = [-100] * len(input_ids_list)  # 默认 label 为 -100
#     labels += target_ids_list
#     labels += [-100] * padding_length

#     # attention_mask：input + target 部分为 1，padding 部分为 0
#     attention_mask = [1] * (len(input_ids_list) + len(target_ids_list)) + [0] * padding_length

#     return {
#         "input_ids": final_input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

#     # attention = ret["attention_mask"]
#     # first_non_neg100 = None
#     # first_neg100_after = None

#     # # 找到第一个不是 -100 的下标
#     # for i, num in enumerate(labels):
#     #     if num != -100:
#     #         first_non_neg100 = i
#     #         break

#     # # 从 first_non_neg100 之后找到第一个 -100 的下标
#     # if first_non_neg100 is not None:
#     #     for i in range(first_non_neg100 + 1, len(labels)):
#     #         if labels[i] == -100:
#     #             first_neg100_after = i
#     #             break

#     # print("第一个不是 -100 的下标:", first_non_neg100)
#     # print("之后第一个 -100 的下标:", first_neg100_after)

#     # # 2. 处理 attention
#     # first_zero = None

#     # # 找到第一个 0 的下标
#     # for i, num in enumerate(attention):
#     #     if num == 0:
#     #         first_zero = i
#     #         break

#     # print("第一个 0 的下标:", first_zero)
#     # # if first_zero is None:
#     # #     with open("./tmp/attention_mask.json", "w") as f:
#     # #         json.dump(ret, f, indent=4)

#     # print("="*80)

#     # return ret


# tokenized_datasets = dataset.map(preprocess_function, batched=False)

# # ======== 训练参数配置 ========
# training_args = TrainingArguments(
#     output_dir="/home/zhengzhe/TM_SFT/result",
#     eval_strategy="no",                # 训练期间不进行评估
#     learning_rate=1e-4,                     # QLoRA 训练时学习率可以适当提高
#     per_device_train_batch_size=1,          # 4-bit 量化允许 batch_size 稍大
#     gradient_accumulation_steps=2,          # 累积 2 个 batch 再进行一次反向传播
#     num_train_epochs=1,                     # 训练 1 轮
#     save_total_limit=2,
#     save_steps=2000,
#     fp16=False,                              # 启用 FP16 加速
#     logging_dir="/home/zhengzhe/TM_SFT/logs",
#     logging_steps=10,
#     optim="paged_adamw_8bit",               # 低显存优化器
#     label_names=["labels"]
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
#     processing_class=tokenizer
# )

# # ======== 训练 ========
# trainer.train()




# # from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# # # 加载预训练模型和Tokenizer
# # model_name = "/run/determined/NAS1/public/Qwen2.5-7B-Instruct"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForCausalLM.from_pretrained(model_name)

# # import json
# # from datasets import Dataset

# # # 读取 JSONL 文件并提取 messages 字段
# # data = []
# # with open('/home/zhengzhe/high_quality_action.jsonl', 'r', encoding='utf-8') as f:
# #     data = json.load(f)["messages"]

# # # 转换为 Hugging Face Dataset
# # dataset = Dataset.from_list(data)

# # def preprocess_function(examples):
# #     # 拼接 instruction 和 input
# #     input_text = f"{examples['instruction']}{examples['input']}"
    
# #     # 对拼接后的文本进行 Tokenization
# #     return tokenizer(input_text, truncation=True, padding='max_length', max_length=2048)

# # tokenized_datasets = dataset.map(preprocess_function, batched=False)

# # # import json
# # # with open("datasets.json", "w") as f:
# # #     json.dump(tokenized_datasets, f, indent = 4)

# # training_args = TrainingArguments(
# #     output_dir="/home/zhengzhe/SFT/result",  # 模型保存路径
# #     eval_strategy="no",                      # 禁用验证，减少显存占用
# #     learning_rate=2e-5,                      # 学习率
# #     per_device_train_batch_size=1,           # 每张 GPU 的 batch_size
# #     per_device_eval_batch_size=1,            # 每张 GPU 的 batch_size
# #     gradient_accumulation_steps=4,           # 梯度累积步数
# #     num_train_epochs=1,                      # 训练周期数
# #     weight_decay=0.01,                       # 权重衰减
# #     save_total_limit=2,                      # 最多保存的模型检查点数量
# #     save_steps=2000,                         # 每隔 2000 步保存一次模型
# #     fp16=True,                               # 启用混合精度训练
# #     logging_dir="/home/zhengzhe/SFT/logs",   # 日志保存路径
# #     logging_steps=10,                        # 每隔 10 步记录一次日志
# # )
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=tokenized_datasets,
# #     processing_class=tokenizer,
# # )

# # trainer.train()
# # # model.save_pretrained("./fine-tuned-model")
# # # tokenizer.save_pretrained("./fine-tuned-model") 




# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
# from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# import torch
# import json
# from datasets import Dataset
# import bitsandbytes as bnb

# # ======== 加载 Qwen2.5-7B-Instruct（4-bit 量化） ========
# model_name = "/home/zhengzhe/qwen2.5-7B-Instruct"
# # model_name = "/home/zhengzhe/SFT/2epoch_result/first_stage/checkpoint-5396"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

# tokenizer.padding_side = "right"
# tokenizer.pad_token_id = tokenizer.eos_token_id

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",               # 推荐用于训练
#     bnb_4bit_use_double_quant=True,          # 双量化进一步节省显存
#     bnb_4bit_compute_dtype=torch.bfloat16    # 计算使用 BF16 提速
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto"   # 自动分配到 GPU
# )
# model = prepare_model_for_kbit_training(model)

# model.config.use_cache = False
# model.gradient_checkpointing_enable()

# # ======== LoRA 配置 ========
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=16,                # LoRA 矩阵秩
#     lora_alpha=32,       # LoRA scaling factor
#     lora_dropout=0.05,   # Dropout 防止过拟合
#     target_modules=["q_proj", "v_proj"],  # 只训练注意力层的 Q/V 权重
#     use_dora = False
# )

# # 将 LoRA 适配到模型nx
# model = get_peft_model(model, lora_config)

# # ======== 读取 JSONL 数据 ========

# with open('/home/zhengzhe/dataset/base_agent_clean_dataset.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# dataset = Dataset.from_list(data)

# import random

# def preprocess_function(examples):
#     input_text = examples['input']
#     target_text = examples['output']

#     # 仅进行截断，不进行填充
#     input_ids = tokenizer(input_text, truncation=True, padding=False)
#     target_ids = tokenizer(target_text, truncation=True, padding=False)

#     # 获取实际的 token ID 序列
#     input_ids_list = input_ids["input_ids"]
#     target_ids_list = target_ids["input_ids"]
#     target_ids_list.append(tokenizer.eos_token_id)

#     # 统一填充到 max_length（保证 batch 训练对齐）
#     max_length = 12 * 1024
#     # 限制 input + target 总长度不超过 max_length
#     max_input_length = max_length - len(target_ids_list)  # 预留 output 空间
#     if len(input_ids_list) > max_input_length:
#         input_ids_list = input_ids_list[:max_input_length]  # 截断 input
    
#     # 合并 input 和 target
#     final_input_ids = input_ids_list + target_ids_list  # input 和 target 合并

#     # 计算 padding 长度
#     padding_length = max_length - len(final_input_ids)
#     final_input_ids += [tokenizer.pad_token_id] * padding_length

#     labels = [-100] * len(input_ids_list)  # 默认 label 为 -100
#     labels += target_ids_list
#     labels += [-100] * padding_length

#     # attention_mask：input + target 部分为 1，padding 部分为 0
#     attention_mask = [1] * (len(input_ids_list) + len(target_ids_list)) + [0] * padding_length

#     return {
#         "input_ids": final_input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

# tokenized_datasets = dataset.map(preprocess_function, batched=False)

# # ======== 训练参数配置 ========
# training_args = TrainingArguments(
#     output_dir="/home/zhengzhe/SFT/lora_clean_result/",
#     eval_strategy="no",                # 训练期间不进行评估
#     learning_rate=5e-5,                     # QLoRA 训练时学习率可以适当提高
#     per_device_train_batch_size=1,          # 4-bit 量化允许 batch_size 稍大
#     gradient_accumulation_steps=1,          # 累积 2 个 batch 再进行一次反向传播
#     num_train_epochs=1,                     # 训练 1 轮
#     save_total_limit=2,
#     save_steps=8000,

#     warmup_steps=50,
#     weight_decay=0.01,
#     lr_scheduler_type="cosine",

#     fp16=False,                              # 启用 FP16 加速
#     bf16=True,
#     logging_dir="/home/zhengzhe/SFT/logs",
#     logging_steps=10,
#     optim="adamw_torch_fused",               # 低显存优化器
#     label_names=["labels"]
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
#     processing_class=tokenizer,
#     optimizers=(None, None)
# )

# # ======== 训练 ========
# trainer.train()



from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
import json
from datasets import Dataset
import bitsandbytes as bnb

# ======== 加载 Qwen2.5-7B-Instruct（4-bit 量化） ========
model_name = "/home/zhengzhe/qwen2.5-7B-Instruct"
# model_name = "/home/zhengzhe/SFT/dedup_2epoch_result/first_stage_deploy_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",               # 推荐用于训练
#     bnb_4bit_use_double_quant=True,          # 双量化进一步节省显存
#     bnb_4bit_compute_dtype=torch.bfloat16    # 计算使用 BF16 提速
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
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

# 将 LoRA 适配到模型nx
model = get_peft_model(model, lora_config)

# ======== 读取 JSONL 数据 ========

with open('/home/zhengzhe/dataset/base_agent_dedup_dataset.json', 'r', encoding='utf-8') as f:
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
    output_dir="/home/zhengzhe/SFT/dedup_lora_result/",
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
    logging_dir="/home/zhengzhe/SFT/logs",
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