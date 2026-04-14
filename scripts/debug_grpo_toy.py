import argparse
import copy
import json
import random
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")

from src.training.grpo_trainer import GRPOConfig, GRPOTrainer


class ToyPromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


class RewardOutput:
    def __init__(self, rewards):
        self.rewards = rewards


class ToyRewardModel:
    def __init__(self, prompt_to_target):
        self.prompt_to_target = prompt_to_target

    def compute_reward(self, images, prompts):
        rewards = []
        for img, prompt in zip(images, prompts):
            target = self.prompt_to_target[prompt]
            reward = 1.0 if img == target else -1.0
            rewards.append(reward)
        return RewardOutput(torch.tensor(rewards, dtype=torch.float32))


class TinyPolicy(nn.Module):
    def __init__(self, num_prompts, vocab_size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_prompts, vocab_size))


class ToyGenerator:
    def __init__(self, prompts, vocab_size=4):
        self.prompt_to_idx = {p: i for i, p in enumerate(prompts)}
        self.vocab_size = vocab_size
        self.device = "cpu"
        self.model = TinyPolicy(len(prompts), vocab_size)
        self.lora_enabled = True

    def get_trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def generate_with_logprobs(self, prompt, return_generation_info=False, **kwargs):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        sampled = []
        log_probs = []
        generation_info = []

        for p in prompts:
            idx = self.prompt_to_idx[p]
            logits = self.model.logits[idx]
            dist = torch.distributions.Categorical(logits=logits)
            token = dist.sample()
            sampled.append(int(token.item()))
            log_probs.append(dist.log_prob(token))
            generation_info.append({"prompt_idx": idx, "token": int(token.item())})

        log_probs = torch.stack(log_probs)
        if return_generation_info:
            return sampled, log_probs, generation_info
        return sampled, log_probs

    def score_from_generation_info(self, generation_info, model=None, use_grad=False):
        score_model = model or self.model
        scores = []
        for item in generation_info:
            idx = item["prompt_idx"]
            token = item["token"]
            dist = torch.distributions.Categorical(logits=score_model.logits[idx])
            scores.append(dist.log_prob(torch.tensor(token)))
        return torch.stack(scores)


def collate_fn(batch):
    return {"prompt": [x["prompt"] for x in batch]}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(kl_coef, steps, seed, ppo_epochs):
    set_seed(seed)

    prompts = [f"prompt_{i}" for i in range(8)]
    prompt_to_target = {p: (i % 4) for i, p in enumerate(prompts)}

    dataset = ToyPromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    generator = ToyGenerator(prompts)
    reward_model = ToyRewardModel(prompt_to_target)

    cfg = GRPOConfig(
        learning_rate=0.2,
        num_epochs=999,
        batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1000,
        save_steps=100000,
        eval_steps=100000,
        output_dir="./outputs/toy_grpo",
        use_wandb=False,
        num_samples_per_prompt=4,
        kl_coef=kl_coef,
        ppo_epochs=ppo_epochs,
        baseline_type="mean",
        use_advantage_normalization=True,
        warmup_steps=0,
    )

    trainer = GRPOTrainer(
        generator=generator,
        reward_model=reward_model,
        train_dataloader=dataloader,
        grpo_config=cfg,
    )

    metrics = []
    iterator = iter(dataloader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        rollout = trainer._prepare_rollout_batch(batch)
        loss_dict = None
        for _ in range(ppo_epochs):
            loss_dict = trainer._compute_replay_loss(rollout)
            loss = loss_dict["loss"]
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()

        with torch.no_grad():
            avg_correct_prob = 0.0
            for prompt in prompts:
                idx = generator.prompt_to_idx[prompt]
                probs = torch.softmax(generator.model.logits[idx], dim=-1)
                avg_correct_prob += probs[prompt_to_target[prompt]].item()
            avg_correct_prob /= len(prompts)

        step_metrics = {
            "step": step + 1,
            "loss": float(loss_dict["loss"].item()),
            "policy_loss": float(loss_dict["policy_loss"].item()),
            "kl_div": float(loss_dict["kl_div"].item()),
            "reward_mean": float(loss_dict["reward_mean"].item()),
            "reward_std": float(loss_dict["reward_std"].item()),
            "ratio_mean": float(loss_dict.get("ratio_mean", torch.tensor(1.0)).item()),
            "clip_fraction": float(loss_dict.get("clip_fraction", torch.tensor(0.0)).item()),
            "avg_correct_prob": avg_correct_prob,
        }
        metrics.append(step_metrics)

    return {
        "config": {"kl_coef": kl_coef, "steps": steps, "seed": seed, "ppo_epochs": ppo_epochs},
        "first": metrics[:5],
        "last": metrics[-5:],
        "final": metrics[-1],
        "history": metrics,
        "best_correct_prob": max(m["avg_correct_prob"] for m in metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl-values", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument("--ppo-epochs", type=int, default=1)
    args = parser.parse_args()

    results = {
        f"kl_{kl}": run_experiment(kl, args.steps, args.seed, args.ppo_epochs)
        for kl in args.kl_values
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
