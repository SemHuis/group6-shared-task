import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any

class LogicWeightedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        alpha: 结论段权重 (l_class) - 包含 "Validity Conclusion:" 及其结果
        beta:  符号段权重 (l_conv)  - 包含 "<think>" 及其符号化过程
        gamma: 规则段权重 (l_rule)  - 包含 "Step-by-Step Derivation" 及其规则检查
        """
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        
        # 锚点配置：已移除 mid 的冒号以增加分词鲁棒性
        self.anchors = {
            "start": "<think>", 
            "mid": "Step-by-Step Derivation", 
            "end": "Validity Conclusion:"
        }
        self.anchor_ids = {}
        self.current_losses = {"loss_a": 0.0, "loss_b": 0.0, "loss_c": 0.0}

    def _prepare_anchors(self, tokenizer: Any):
        for key, text in self.anchors.items():
            # add_special_tokens=False 确保匹配的是文本核心 ID
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.anchor_ids[key] = torch.tensor(ids, device="cuda")

    def _find_subsequence_ids(self, source, target):
        s_len, t_len = source.size(0), target.size(0)
        if t_len == 0 or t_len > s_len: return -1
        for i in range(s_len - t_len + 1):
            if torch.equal(source[i : i + t_len], target):
                return i
        return -1

    def forward(self, logits, labels, **kwargs):
        # 1. 移位对齐 (Shift for Causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 2. 惰性初始化锚点 ID
        if not self.anchor_ids and "tokenizer" in kwargs:
            self._prepare_anchors(kwargs["tokenizer"])
            
        batch_size, seq_len, vocab_size = shift_logits.shape
        
        # 3. 计算基础 Token Loss
        per_token_loss = self.ce_loss(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        ).view(batch_size, seq_len)
        
        total_loss = torch.tensor(0.0, device=logits.device)
        valid_samples = 0
        tmp_a, tmp_b, tmp_c = [], [], []

        for b in range(batch_size):
            label_seq = shift_labels[b]
            loss_seq = per_token_loss[b]
            
            # 4. 寻找锚点索引
            idx_start = self._find_subsequence_ids(label_seq, self.anchor_ids.get("start", torch.tensor([])))
            idx_mid = self._find_subsequence_ids(label_seq, self.anchor_ids.get("mid", torch.tensor([])))
            idx_end = self._find_subsequence_ids(label_seq, self.anchor_ids.get("end", torch.tensor([])))
            
            # 5. 保底逻辑：若任一锚点缺失，退化为标准平均 Loss
            if idx_start == -1 or idx_mid == -1 or idx_end == -1:
                valid_mask = (label_seq != -100)
                if valid_mask.any():
                    total_loss += loss_seq[valid_mask].mean()
                    valid_samples += 1
                continue

            # 寻找序列有效结尾
            valid_idx = (label_seq != -100).nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0: continue
            last_idx = valid_idx[-1].item()
            
            def get_mean(s, e): 
                return loss_seq[s:e].mean() if e > s else torch.tensor(0.0, device=logits.device)

            # 【核心修复】：起始索引不再增加 len(anchor)，确保锚点本身参与梯度计算
            # 这样模型必须学会在正确的位置输出模板字符串
            l_conv = get_mean(idx_start, idx_mid)
            l_rule = get_mean(idx_mid, idx_end)
            l_class = get_mean(idx_end, last_idx + 1)

            # 7. 加权求和 (按你之前定义的 alpha/beta/gamma 映射)
            sample_loss = self.alpha * l_class + self.beta * l_conv + self.gamma * l_rule
            total_loss += sample_loss
            
            tmp_a.append(l_conv.item())
            tmp_b.append(l_rule.item())
            tmp_c.append(l_class.item())
            valid_samples += 1

        # 8. 日志统计
        if valid_samples > 0:
            la = sum(tmp_a)/len(tmp_a) if tmp_a else 0.0
            lb = sum(tmp_b)/len(tmp_b) if tmp_b else 0.0
            lc = sum(tmp_c)/len(tmp_c) if tmp_c else 0.0
            
            if dist.is_initialized():
                t = torch.tensor([la, lb, lc], device=logits.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= dist.get_world_size()
                la, lb, lc = t.tolist()
            self.current_losses = {"loss_a": la, "loss_b": lb, "loss_c": lc}
            
        return total_loss / max(valid_samples, 1)