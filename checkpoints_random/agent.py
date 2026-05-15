from main_env import CinchMainEnv

import torch
import torch.nn as nn

# Make a dummy agent that just returns random legal actions. This is used to test the get_agent function and the checkpoint loading.
class CinchDummyAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, legal):
        # Dummy forward pass - replace with actual implementation
        logits = torch.randn(52)  # Random logits for 52 cards
        mask = torch.full_like(logits, -1e9)
        mask[legal] = 0.0
        logits = logits + mask
        return logits, torch.randn(1).squeeze(-1)

def create_agent(env: CinchMainEnv) -> CinchDummyAgent:
    """
    Creates a new instance of the CinchDummyAgent.

    Args:
        env: The CinchMainEnv instance.
    
    Returns:
        An instance of the CinchDummyAgent.
    """
    return CinchDummyAgent()

def select_action(model: CinchDummyAgent, obs: dict, env: CinchMainEnv, DEVICE: torch.device) -> int:
    """
    Selects an action using the model's forward pass.
    
    Args:
        model: An instance of the CinchDummyAgent.
        obs: The current observation from the environment.
        env: The environment instance, used to get legal actions and other info.
        DEVICE: The device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        The index of the selected action.
    """
    legal = env.legal_actions()

    with torch.no_grad():
        logits, _ = model(legal)

    # Print the probabilities of the legal actions for debugging
    # probs = torch.softmax(logits, dim=-1)
    # legal_probs = {env._index_to_card(a): probs[a].item() for a in legal}
    # print("Legal action probabilities:", {f"{rank}{suit}": prob for (suit, rank), prob in legal_probs.items()})

    return torch.argmax(logits).item()