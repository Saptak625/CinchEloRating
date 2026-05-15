import os
import importlib.util
import torch

from main_env import CinchMainEnv, SUITS, RANKS

DEVICE = "cpu"

def get_agent(env: CinchMainEnv, checkpoint_path: str):
    # Call the create_agent function from agent.py in the checkpoint_path directory.
    directory = os.path.dirname(checkpoint_path)
    spec = importlib.util.spec_from_file_location("agent", os.path.join(directory, "agent.py"))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    model = agent_module.create_agent(env).to(DEVICE)
    # print(f'Found {type(model).__name__} with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Load the checkpoint and set the model weights.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # print(f'Loaded checkpoint from {checkpoint_path}')
    return model

def select_action(model: torch.nn.Module, checkpoint_path: str, obs: dict, env: CinchMainEnv) -> int:
    # Call the select_action function from agent.py in the checkpoint_path directory.
    directory = os.path.dirname(checkpoint_path)
    spec = importlib.util.spec_from_file_location("agent", os.path.join(directory, "agent.py"))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    return agent_module.select_action(model, obs, env, DEVICE)


if __name__ == "__main__":
    env = CinchMainEnv()
    checkpoint_path = os.path.join("checkpoints_transformer_small_batches", "ckpt_5000.pt")
    model = get_agent(env, checkpoint_path)
    print(model)

    # Test the select_action function with a dummy observation.
    obs = env.reset()
    action = select_action(model, checkpoint_path, obs, env)
    print('Action selected:', action)
