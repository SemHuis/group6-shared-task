# import torch
# from transformers import AutoTokenizer
# from logic_loss import LogicWeightedLoss

# def test_loss_partitioning():
#     # 1. Setup (using Qwen-14B tokenizer for consistency)
#     model_path = "Qwen/Qwen-14B" # Replace with your local Habrok path
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
#     # 2. Mock a typical Syllogism sequence
#     # Interval A: Conversion, B: Rule Check, C: Verdict
#     text = (
#         "<think> Converting premises to standard form. "
#         "Step-by-Step Derivation: Major term is distributed in conclusion but not in premise. "
#         "Validity Conclusion: Invalid (Illicit Major)."
#     )
    
#     # 3. Tokenize
#     inputs = tokenizer(text, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     labels = input_ids.clone() # Mock SFT labels where response is not masked
    
#     # 4. Initialize Loss
#     criterion = LogicWeightedLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
#     # 5. Mock Logits (batch=1, seq_len, vocab_size)
#     vocab_size = tokenizer.vocab_size
#     mock_logits = torch.randn(1, input_ids.size(1), vocab_size)

#     # 6. Run Forward (passing tokenizer via kwargs)
#     # We shift internally in logic_loss.py
#     loss = criterion(mock_logits, labels, tokenizer=tokenizer)
    
#     # 7. Print Anchor ID checks
#     print("--- Anchor Detection Check ---")
#     for key, ids in criterion.anchor_ids.items():
#         decoded = tokenizer.decode(ids)
#         print(f"Anchor [{key}]: IDs {ids.tolist()} -> '{decoded}'")

#     print("\nLoss value:", loss.item())
#     print("Partitioning validation successful if no index errors occurred.")

# if __name__ == "__main__":
#     test_loss_partitioning()