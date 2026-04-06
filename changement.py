import torch

# modifier trophy_baseline manuelement

ckpt = torch.load("checkpoints/mspacman_dqn.pth", map_location="cpu")
ckpt["trophy_baseline"] = 1200.0  # 120 dots × 10
torch.save(ckpt, "checkpoints/mspacman_dqn.pth")
print("Done:", ckpt["trophy_baseline"])