import torch
import torch.nn as nn
import torch.nn.functional as F

from main_env import CinchMainEnv
from obs_utils import build_mlp_obs

class ResidualBlock(nn.Module):
    def __init__(self, size):
        """
        A simple residual block with two fully connected layers and a skip connection.
        
        Args:
            size: The size of the input and output tensors for the block.
        """
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        """
        A simple residual block with two fully connected layers and a skip connection.
        
        Args:
            x: Input tensor of shape (batch_size, size)
        Returns:
            Output tensor of shape (batch_size, size)
        """
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(self.norm(x + h))


class CinchAgent(nn.Module):
    def __init__(self, input_size, hidden=512):
        """
        A simple feedforward neural network with residual connections for the Cinch agent.
        
        Args:
            input_size: The size of the input tensor (observation space).
            hidden: The size of the hidden layers.
        
        The network architecture is as follows:
        - Fully connected layer from input_size to hidden, followed by ReLU activation
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Policy head: fully connected layer from hidden to 52 (action logits)
        - Value head: fully connected layer from hidden to 1 (state value)
        """
        super().__init__()

        self.fc_in = nn.Linear(input_size, hidden)

        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.res3 = ResidualBlock(hidden)

        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 52)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, legal):
        """
        Forward pass through the network.

        Network architecture:
        - Fully connected layer from input_size to hidden, followed by ReLU activation
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Residual block with two fully connected layers of size hidden
        - Policy head: 
            - Fully connected layer from hidden to hidden, followed by ReLU activation
            - Fully connected layer from hidden to 52 (action logits)
        - Value head:
            - Fully connected layer from hidden to hidden, followed by ReLU activation
            - Fully connected layer from hidden to 1 (state value)
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            logits: Output tensor of shape (batch_size, 52) representing the action logits
            value: Output tensor of shape (batch_size, 1) representing the state value
        """
        x = F.relu(self.fc_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        logits = self.policy(x)
        value = self.value(x).squeeze(-1)

        # Mask illegal actions
        mask = torch.full_like(logits, -1e9)
        mask[legal] = 0.0
        logits = logits + mask

        return logits, value
    

def create_agent(env: CinchMainEnv) -> CinchAgent:
    """
    Creates a new instance of the CinchAgent with the specified input size.

    Args:
        input_size: The size of the input tensor (observation space).
    
    Returns:
        An instance of the CinchAgent.
    """
    input_size = len(build_mlp_obs(env.reset()))
    return CinchAgent(input_size)


def select_action(model: CinchAgent, obs: dict, env: CinchMainEnv, DEVICE: torch.device) -> int:
    """
    Selects an action for the given observation using the provided model.
    
    Args:
        model: An instance of the CinchAgent.
        obs: The current observation from the environment.
        env: The environment instance, used to get legal actions and other info.
        DEVICE: The device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        The index of the selected action.
    """
    legal = env.legal_actions()
    x = build_mlp_obs(obs).to(DEVICE)  # Add batch dimension and move to device

    with torch.no_grad():
        logits, _ = model(x, legal)

    # Print the probabilities of the legal actions for debugging
    # probs = torch.softmax(logits, dim=-1)
    # legal_probs = {env._index_to_card(a): probs[a].item() for a in legal}
    # print("Legal action probabilities:", {f"{rank}{suit}": prob for (suit, rank), prob in legal_probs.items()})

    return torch.argmax(logits).item()