import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import json


def debug_print(msg, obj=None, level="INFO"):
    """统一的调试打印函数"""
    prefix = f"[DEBUG-{level}]"
    if obj is not None:
        if hasattr(obj, 'keys'):
            print(f"{prefix} {msg}: {list(obj.keys())}")
        elif hasattr(obj, 'column_names'):
            print(f"{prefix} {msg}: {obj.column_names}")
        else:
            print(f"{prefix} {msg}: {obj}")
    else:
        print(f"{prefix} {msg}")


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, regularization_weight=0.005, **kwargs):
        train_dataset = kwargs.get('train_dataset')
        self._noise_prompts = None
        if train_dataset is not None and "noise_prompt" in train_dataset.column_names:
            self._noise_prompts = train_dataset["noise_prompt"]
            
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        if self.tokenizer.pad_token_id is not None:
            self.padding_value = self.tokenizer.pad_token_id
        else:
            self.padding_value = 0
        
        # 🔴 只添加attention_mask，不做其他修改
        self._add_attention_masks_only()
        
        self.regularization_weight = regularization_weight
        
        # 添加noise_prompt
        if train_dataset is not None and "noise_prompt" in train_dataset.column_names:
            self._add_noise_prompt_to_dataset()
        
        self.debug_step = 0
    
    def _add_attention_masks_only(self):
        """只添加attention_mask字段，不修改input_ids，不添加labels"""
        
        print("\n" + "="*80)
        print("【_add_attention_masks_only】添加attention_mask")
        print("="*80)
        
        def add_masks(examples):
            """为每个example添加attention_mask"""
            batch_size = len(examples["prompt_input_ids"])
            
            # 为prompt生成attention_mask
            prompt_attention_masks = []
            for prompt_ids in examples["prompt_input_ids"]:
                mask = [1] * len(prompt_ids)
                prompt_attention_masks.append(mask)
            
            # 为chosen生成attention_mask（chosen是response-only）
            chosen_attention_masks = []
            for chosen_ids in examples["chosen_input_ids"]:
                mask = [1] * len(chosen_ids)
                chosen_attention_masks.append(mask)
            
            # 为rejected生成attention_mask（rejected是response-only）
            rejected_attention_masks = []
            for rejected_ids in examples["rejected_input_ids"]:
                mask = [1] * len(rejected_ids)
                rejected_attention_masks.append(mask)
            
            return {
                "prompt_attention_mask": prompt_attention_masks,
                "chosen_attention_mask": chosen_attention_masks,
                "rejected_attention_mask": rejected_attention_masks,
            }
        
        # 检查是否已经有这些字段
        sample = self.train_dataset[0]
        if "chosen_attention_mask" not in sample:
            self.train_dataset = self.train_dataset.map(
                add_masks,
                batched=True,
                desc="Adding attention masks",
            )
            print(f"✅ attention_mask添加完成")
        else:
            print(f"✅ attention_mask已存在，跳过")
        
        print("="*80 + "\n")
    
    def _add_noise_prompt_to_dataset(self):
        """Tokenize noise_prompt并添加到数据集"""
        
        noise_prompts_data = self._noise_prompts
        tokenizer = self.tokenizer
        max_prompt_length = self.max_prompt_length
        
        def tokenize_noise_prompt(examples, indices):
            noise_input_ids_list = []
            noise_attention_mask_list = []
            
            for idx in indices:
                raw_noise_text = noise_prompts_data[idx]
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": raw_noise_text}
                ]
                
                formatted_noise_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                noise_tokens = tokenizer(
                    formatted_noise_prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_length if max_prompt_length else 8*1024,
                )
                
                noise_input_ids_list.append(noise_tokens["input_ids"])
                noise_attention_mask_list.append(noise_tokens["attention_mask"])
            
            return {
                "noise_prompt_input_ids": noise_input_ids_list,
                "noise_prompt_attention_mask": noise_attention_mask_list,
            }
        
        self.train_dataset = self.train_dataset.map(
            tokenize_noise_prompt,
            batched=True,
            with_indices=True,
            desc="Tokenizing noise prompts",
        )
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[list, torch.LongTensor]],
        train_eval: str = "train",
    ):

        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # ========== 1. 原始batch的forward ==========
        forward_output = self.concatenated_forward(model, batch)

        if isinstance(forward_output, dict):
            policy_chosen_logps = forward_output["chosen_logps"]
            policy_rejected_logps = forward_output["rejected_logps"]
            policy_chosen_logits = forward_output.get("mean_chosen_logits", None)
            policy_rejected_logits = forward_output.get("mean_rejected_logits", None)
        else:
            policy_chosen_logps = forward_output[0]
            policy_rejected_logps = forward_output[1]
            policy_chosen_logits = None
            policy_rejected_logits = None
        
        del forward_output
        torch.cuda.empty_cache()
        
        # ========== 2. Reference model ==========
        with torch.no_grad():
            if self.ref_model is not None:
                ref_forward_output = self.concatenated_forward(self.ref_model, batch)
                
                if isinstance(ref_forward_output, dict):
                    ref_chosen_logps = ref_forward_output["chosen_logps"]
                    ref_rejected_logps = ref_forward_output["rejected_logps"]
                else:
                    ref_chosen_logps = ref_forward_output[0]
                    ref_rejected_logps = ref_forward_output[1]
                
            else:
                base = model.module if hasattr(model, "module") else model
                # ✅ 正确做法：临时禁用adapter获取base model输出
                with base.disable_adapter():
                    ref_forward_output = self.concatenated_forward(base, batch)
                
                if isinstance(ref_forward_output, dict):
                    ref_chosen_logps = ref_forward_output["chosen_logps"]
                    ref_rejected_logps = ref_forward_output["rejected_logps"]
                else:
                    ref_chosen_logps = ref_forward_output[0]
                    ref_rejected_logps = ref_forward_output[1]

            del ref_forward_output
            torch.cuda.empty_cache()


        # ========== 3. DPO loss ==========
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        
        if self.ref_model is not None:
            del ref_chosen_logps, ref_rejected_logps
            torch.cuda.empty_cache()
        
        # ========== 4. 正则项 ==========
        regularization_loss = torch.tensor(0.0, device=losses.device)
        
        has_noise_data = ("noise_prompt_input_ids" in batch and 
                         "noise_prompt_attention_mask" in batch)
        
        if has_noise_data:
            noise_batch = {
                "prompt_input_ids": batch["noise_prompt_input_ids"],
                "prompt_attention_mask": batch["noise_prompt_attention_mask"],
                "chosen_input_ids": batch["chosen_input_ids"],
                "chosen_attention_mask": batch["chosen_attention_mask"],
                "rejected_input_ids": batch["rejected_input_ids"],
                "rejected_attention_mask": batch["rejected_attention_mask"],
            }
            
            noise_forward_output = self.concatenated_forward(model, noise_batch)
            
            if isinstance(noise_forward_output, dict):
                noise_chosen_logps = noise_forward_output["chosen_logps"]
                noise_rejected_logps = noise_forward_output["rejected_logps"]
            else:
                noise_chosen_logps = noise_forward_output[0]
                noise_rejected_logps = noise_forward_output[1]
            
            del noise_forward_output, noise_batch
            torch.cuda.empty_cache()
            
            # 🔴 计算advantage
            advantage_original = (policy_chosen_logps - policy_rejected_logps).detach()
            advantage_noise = noise_chosen_logps - noise_rejected_logps
            
            # Clipped L2 Loss
            advantage_diff = advantage_noise - advantage_original
            advantage_diff_clipped = torch.clamp(advantage_diff, min=-10, max=10)  # 截断极端值

            regularization_loss = F.mse_loss(
                advantage_diff_clipped,
                torch.zeros_like(advantage_diff_clipped),
                reduction='mean'
            )
            
            metrics[f"{prefix}regularization"] = regularization_loss.detach().cpu().item()
            metrics[f"{prefix}advantage_original"] = advantage_original.mean().detach().cpu().item()
            metrics[f"{prefix}advantage_noise"] = advantage_noise.mean().detach().cpu().item()
            metrics[f"{prefix}advantage_diff_abs"] = advantage_diff.abs().mean().detach().cpu().item()
            
            del noise_chosen_logps, noise_rejected_logps, advantage_noise

        
        # ========== 5. 总loss ==========
        dpo_loss_mean = losses.mean()
        total_loss = dpo_loss_mean + self.regularization_weight * regularization_loss
        
        # ========== 6. Metrics ==========
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}loss/dpo"] = dpo_loss_mean.detach().cpu()
        metrics[f"{prefix}loss/total"] = total_loss.detach().cpu()
        
        if policy_rejected_logits is not None:
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        if policy_chosen_logits is not None:
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        self.debug_step += 1

        return total_loss, metrics

    def _create_noise_batch(self, batch: Dict) -> Dict:
        """创建noise batch - 使用DPO标准格式"""
        
        noise_batch = {
            "prompt_input_ids": batch["noise_prompt_input_ids"],  # noise作为新的prompt
            "prompt_attention_mask": batch["noise_prompt_attention_mask"],
            # 🔴 chosen/rejected仍然只是response部分，不包含prompt
            "chosen_input_ids": batch["chosen_input_ids"],  # 已经是response-only
            "chosen_attention_mask": batch["chosen_attention_mask"],
            "rejected_input_ids": batch["rejected_input_ids"],
            "rejected_attention_mask": batch["rejected_attention_mask"],
        }
        
        # 🔴 不需要手动拼接，concatenated_forward会处理
        
        return noise_batch

    
    def _pad_tensors(self, tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        """对tensor列表进行padding"""
        max_len = max(t.shape[0] for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype
        
        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad_size = max_len - t.shape[0]
                t = torch.cat([t, torch.full((pad_size,), padding_value, dtype=dtype, device=device)])
            padded.append(t)
        
        return torch.stack(padded)


class CustomDPODataCollator(DPODataCollatorWithPadding):
    """
    继承DPODataCollatorWithPadding，添加noise_prompt处理
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 调用父类collator
        batch = super().__call__(features)
        
        # 只添加noise_prompt字段
        if "noise_prompt_input_ids" in features[0]:
            max_len = max(len(f["noise_prompt_input_ids"]) for f in features)
            
            padded_noise_ids = []
            padded_noise_mask = []
            
            for f in features:
                ids = f["noise_prompt_input_ids"]
                mask = f["noise_prompt_attention_mask"]
                
                padded_ids = ids + [self.pad_token_id] * (max_len - len(ids))
                padded_mask = mask + [0] * (max_len - len(mask))
                
                padded_noise_ids.append(padded_ids)
                padded_noise_mask.append(padded_mask)
            
            batch["noise_prompt_input_ids"] = torch.tensor(padded_noise_ids, dtype=torch.long)
            batch["noise_prompt_attention_mask"] = torch.tensor(padded_noise_mask, dtype=torch.long)
        
        return batch
