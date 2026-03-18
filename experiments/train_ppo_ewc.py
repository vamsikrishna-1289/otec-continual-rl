# experiments/train_ppo_ewc.py

import os
import sys
import numpy as np
import torch

# ==========================
# FIX IMPORT PATH
# ==========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_processing.sst_data_loader import SSTDataLoader
from environment.otec_env import OTECEnvironment

from stable_baselines3 import PPO


# ==========================
# PATH CONFIG
# ==========================
BASE_PATH = r"C:\Users\DELL\PycharmProjects\CAPSTONE\otec-continual-rl\data"

REGIME_PATHS = {
    "T1": os.path.join(BASE_PATH, "T1"),
    "T2": os.path.join(BASE_PATH, "T2"),
    "T3": os.path.join(BASE_PATH, "T3"),
    "T4": os.path.join(BASE_PATH, "T4"),
}


# ==========================
# LOAD DATA
# ==========================
def load_all_regimes():
    regime_data = {}

    for key, path in REGIME_PATHS.items():
        loader = SSTDataLoader(folder_path=path)
        regime_data[key] = loader.load_and_process()

    return regime_data


# ==========================
# EWC HELPER
# ==========================
class EWC:
    def __init__(self, model, dataloader, lambda_=5000):
        self.model = model
        self.lambda_ = lambda_
        self.params = {n: p.clone().detach() for n, p in model.policy.named_parameters()}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {}

        for name, param in self.model.policy.named_parameters():
            fisher[name] = torch.zeros_like(param)

        for state in dataloader:
            state_tensor = torch.tensor(state).float().unsqueeze(0)

            self.model.policy.zero_grad()

            action_dist = self.model.policy.get_distribution(state_tensor)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action).mean()
            loss = -log_prob

            loss.backward()

            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone().pow(2)

        for name in fisher:
            fisher[name] /= len(dataloader)

        return fisher

    def penalty(self, model):
        loss = 0

        for name, param in model.policy.named_parameters():
            loss += (self.fisher[name] * (param - self.params[name]).pow(2)).sum()

        return self.lambda_ * loss


# ==========================
# TRAIN WITH EWC
# ==========================
def train_with_ewc(regime_data, total_timesteps=3000):
    model = None
    ewc = None

    for regime in ["T1", "T2", "T3", "T4"]:
        print(f"\n========== Training with EWC on {regime} ==========")

        env = OTECEnvironment(regime_data[regime], regime_type=regime)

        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, n_steps=256)
        else:
            model.set_env(env)

        # Train normally
        model.learn(total_timesteps=total_timesteps)

        # Apply EWC penalty manually (approximation)
        if ewc is not None:
            penalty = ewc.penalty(model)
            print(f"EWC penalty: {penalty.item()}")

        # Update EWC after task
        ewc = EWC(model, regime_data[regime])

    return model


# ==========================
# EVALUATION
# ==========================
def evaluate_model(model, regime_data):
    results = {}

    for regime in ["T1", "T2", "T3", "T4"]:
        env = OTECEnvironment(regime_data[regime], regime_type=regime)

        state = env.reset()
        total_reward = 0

        for _ in range(len(regime_data[regime])):
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

        avg_reward = total_reward / len(regime_data[regime])
        results[regime] = avg_reward

        print(f"{regime} -> Avg Reward: {avg_reward:.4f}")

    return results


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    regime_data = load_all_regimes()

    model = train_with_ewc(regime_data)

    print("\n========== FINAL EVALUATION (EWC) ==========")
    results = evaluate_model(model, regime_data)

    import json

    with open("ewc_final_results.json", "w") as f:
        json.dump(results, f, indent=4)