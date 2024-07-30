# Imports
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from DQN_model import Qnet
from utils import ReplayBuffer, train
from padm_env import TreasureHunt  # Assuming your class is in treasure_hunt.py

# User definitions
train_dqn = True
test_dqn = False
render = False

# Define environment specific attributes
no_actions = 4  # Up, Down, Left, Right

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
buffer_limit = 100_000
batch_size = 256
num_episodes = 2_000
max_steps = 1_000

epsilon_start = 1.0
epsilon_end = 0.5
epsilon_decay = 1_500

# Initialize Environment and Get State Dimensions
env = TreasureHunt(grid_size=5, seed=42)
state, _ = env.reset()
print("State shape:", state.shape)  # This will print the shape of the state

# Set no_states to match the state dimensions
no_states = 2

# Main
if train_dqn:
    # Initialize the Q Net and the Q Target Net
    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn_target = Qnet(no_actions=no_actions, no_states=no_states)
    dqn_target.load_state_dict(dqn.state_dict())
    dqn_target.eval()
    
    try:
        dqn.load_state_dict(torch.load("dqn.pth"))
        dqn.eval()
        print(dqn.state_dict().keys())
    except FileNotFoundError:
        print("Pretrained model not found, starting from scratch.")
    
    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        # Epsilon decay (Please come up with your own logic)
        epsilon = max(epsilon_end, epsilon_start - n_epi / epsilon_decay)/100  # Linear annealing

        s, _ = env.reset()
        done = False

        # Define maximum steps per episode
        for _ in range(max_steps):
            # Choose an action (Exploration vs. Exploitation)
            a = dqn.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            if render:
                env.render()
            done_mask = 0.0 if done else 1.0

            # Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(dqn, dqn_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            dqn_target.load_state_dict(dqn.state_dict())
            print(f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        # Define a stopping condition for the game:
        if len(rewards) >= 10 and np.mean(rewards[-10:]) >= max_steps:
            break

    env.close()

    # Save the trained Q-net
    torch.save(dqn.state_dict(), "dqn.pth")

    # Plot the training curve
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test
if test_dqn:
    print("Testing the trained DQN:")
    
    env = TreasureHunt(grid_size=5, seed=42)  # Your custom environment

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    
    try:
        dqn.load_state_dict(torch.load("dqn.pth"))
    except EOFError:
        print("Error: EOFError - The model file is empty or corrupted.")
    except FileNotFoundError:
        print("Error: FileNotFoundError - The model file is not found.")
    except Exception as e:
        print(f"Error: {e}")
    
    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # Completely exploit
            if render:
                env.render()
            
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, info = env.step(action.argmax().item())
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()