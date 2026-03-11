from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
# 彻底删除这一行: from llamafactory.train.loss import REGISTER_LOSS 

class LogicLogCallback(TrainerCallback):
    """
    Callback to log segmented logic losses.
    """
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Llama-Factory 的 trainer 会将自己传递给 callback
        trainer = kwargs.get("trainer")
        
        # 检查我们的 logic_criterion 是否存在并持有数据
        if trainer and hasattr(trainer, "logic_criterion"):
            cl = trainer.logic_criterion
            logs = {
                "l_conv_A": cl.current_losses["loss_a"],
                "l_rule_B": cl.current_losses["loss_b"],
                "l_verdict_C": cl.current_losses["loss_c"],
            }
            # 同步到 Trainer 的日志状态中
            state.log_history.append({**logs, **{"step": state.global_step}})
            
            # 终端实时反馈
            print(f"\n[Logic Monitor] Step {state.global_step} | "
                  f"A(Conv): {logs['l_conv_A']:.4f} | "
                  f"B(Rule): {logs['l_rule_B']:.4f} | "
                  f"C(Verdict): {logs['l_verdict_C']:.4f}")