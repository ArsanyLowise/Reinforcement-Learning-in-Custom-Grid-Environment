# Import:
# -------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# Deep Q-Network:
class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 128, bias=True)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.fc3 = nn.Linear(128, 2, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def sample_action(self, obs, epsilon):
    #     out = self.forward(obs)
    #     coin = torch.rand(1).item()
    #     if coin < epsilon:
    #         # return torch.randint(0, self.fc3.out_features, (1,)).item()
    #         gym.spaces.Discrete(4).sample()

    #     else:
    #         return out.argmax().item()   
#     def sample_action(self, obs, epsilon):
#         out = self.forward(obs)
#         coin = np.random.rand()
#         if coin < epsilon:
#             return np.random.randint(0, self.fc3.out_features)
#         else:
#             return out.argmax().item()

# no_actions = 2  # Adjust this to match the saved model
# no_states = 4  # Adjust this to match the saved model

    def sample_action(self, obs, epsilon):
            coin = random.random()
            if coin < epsilon:
                return random.randint(0, self.fc3.out_features - 1)
            else:
                out = self.forward(obs)
                return out.argmax().item()

# Usage Example
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    no_states = env.observation_space.shape[0]
    no_actions = env.action_space.n

    q_net = Qnet(no_states, no_actions)

    # Example observation
    obs = torch.tensor(env.reset()[0], dtype=torch.float32)
    epsilon = 0.1

    # Sample an action
    action = q_net.sample_action(obs, epsilon)
    print("Sampled action:", action)