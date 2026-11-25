"""train.step_loss gives the same loss as model(labels=...).loss, through peft, on a tiny random Qwen3.
Needs torch, so it runs in a training venv, not the core suite:

    tools\\train\\.venv-cpu\\Scripts\\python tools\\train\\test_train.py
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import train  # noqa: E402


class TestStepLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
        self.model = Qwen3ForCausalLM(cfg).eval()
        rng = random.Random(1)
        prompt, target = [rng.randrange(1000) for _ in range(90)], [rng.randrange(1000) for _ in range(12)]
        self.ex = {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target, "weight": 1.0}

    def full_loss(self, model):
        ids = torch.tensor([self.ex["input_ids"]])
        return float(model(input_ids=ids, labels=torch.tensor([self.ex["labels"]])).loss)

    def test_same_loss_as_full_logits(self):
        with torch.no_grad():
            want, got = self.full_loss(self.model), float(train.step_loss(self.model, self.ex, "cpu"))
        print(f"\nbase: got {got:.6f} expected {want:.6f}")
        self.assertAlmostEqual(got, want, places=5)

    def test_same_loss_through_peft_with_a_trained_adapter(self):
        m = get_peft_model(self.model, LoraConfig(r=4, lora_alpha=8, target_modules=train.TARGET_MODULES))
        with torch.no_grad():   # B starts at zero; move it so the adapter changes the logits
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)
            want, got = self.full_loss(m), float(train.step_loss(m, self.ex, "cpu"))
        print(f"\npeft: got {got:.6f} expected {want:.6f}")
        self.assertAlmostEqual(got, want, places=5)

    def test_weight_scales_and_gradient_reaches_lora(self):
        m = get_peft_model(self.model, LoraConfig(r=4, lora_alpha=8, target_modules=train.TARGET_MODULES))
        base = float(train.step_loss(m, self.ex, "cpu"))
        loss = train.step_loss(m, {**self.ex, "weight": 0.5}, "cpu")
        loss.backward()
        grads = [p.grad for n, p in m.named_parameters() if "lora_" in n and p.grad is not None]
        print(f"\nweight 0.5: got {float(loss):.6f} expected {base * 0.5:.6f}; lora grads {len(grads)}")
        self.assertAlmostEqual(float(loss), base * 0.5, places=5)
        self.assertTrue(grads and any(float(g.abs().sum()) > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
