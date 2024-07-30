# Importing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to train Q-learning agent
def train_q_learning(env, 
                     no_episodes, 
                     epsilon, epsilon_min, 
                     epsilon_decay, alpha, 
                     gamma, 
                     interrupt_event=None, 
                     q_table_save_path="q_table.npy"):
    # Initialize Q-table
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        total_reward = 0  # Initialize total reward for the episode
        done = False

        while not done:
            # Choose action epsilon-greedily
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # exploration
            else:
                action = np.argmax(q_table[state])  # exploitation

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward
            env.render()

            # Add debugging prints to trace the issue
            print(f"Current state: {state}, Action: {action}, Next state: {next_state}")

            # Check if indices are within bounds
            if (0 <= state[0] < env.grid_size and 0 <= state[1] < env.grid_size and
                0 <= next_state[0] < env.grid_size and 0 <= next_state[1] < env.grid_size and
                0 <= action < env.action_space.n):
                # Update Q-values using Q-learning update rule
                q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
                    alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])
            else:
                print(f"Index out of bounds detected. State: {state}, Action: {action}, Next state: {next_state}")

            state = next_state

            # Check for interrupt condition
            if interrupt_event and interrupt_event(state, done, reward):
                print("Interrupting training due to custom condition.")
                return

        # Perform epsilon decay after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print episode summary
        print(f"Episode {episode + 1}, total reward: {total_reward}")

    # Save Q-table after training completes
    np.save(q_table_save_path, q_table)
    print("Q-table saved.")

    # Close environment after training
    env.close()
    print("Training finished.\n")

# Function to visualize the Q-table

def visualize_q_table(q_values_path="q_table.npy", 
                      hell_state_coordinates=[(2, 1), (0, 4)], 
                      goal_coordinates=(2, 4), 
                      actions=["Up", "Down", "Right", "Left"]):
   # Load the Q_table
    try:
        q_table = np.load(q_values_path)
        print(f"Loaded Q-table shape: {q_table.shape}")

        # Create subplots for each action
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = np.rot90(q_table[:, :, i], k=1)  # Rotate 90 degrees counterclockwise
            mask = np.zeros_like(heatmap_data, dtype=bool)
            goal_rotated = (q_table.shape[0] - 1 - goal_coordinates[1], goal_coordinates[0])
            mask[goal_rotated] = True
            for hell_state in hell_state_coordinates:
                hell_rotated = (q_table.shape[0] - 1 - hell_state[1], hell_state[0])
                mask[hell_rotated] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Denote Goal and Hell states
            ax.text(goal_rotated[1] + 0.5, goal_rotated[0] + 0.5, "G", color="green", ha="center", va="center", weight="bold", fontsize=14)
            for hell_state in hell_state_coordinates:
                hell_rotated = (q_table.shape[0] - 1 - hell_state[1], hell_state[0])
                ax.text(hell_rotated[1] + 0.5, hell_rotated[0] + 0.5, "H", color="red", ha="center", va="center", weight="bold", fontsize=14)

            ax.set_title(f"Action: {action}")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

goal_coordinates = (2, 4)
hell_state_coordinates = [(2, 0), (0, 4)]

# visualize_q_table(hell_state_coordinates=hell_state_coordinates,
#                   goal_coordinates=goal_coordinates,
#                   actions=["Up", "Down", "Right", "Left"],
#                   q_values_path="q_table.npy")