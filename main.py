# Importing necessary libraries
import numpy as np
from padm_env import TreasureHunt  
from q_learning import train_q_learning, visualize_q_table

# User definitions
train = True
visualize_results = True

# Set a random seed for reproducibility
random_seed = 42

# Function to create an instance of the environment
def create_env(goal_coordinates, hell_state_coordinates, obstacles_coordinates, treasures_coordinates, seed=None):
    grid_size = 5  # Increased grid size to accommodate coordinates
    env = TreasureHunt(grid_size=grid_size, seed=seed, goal_coordinates=goal_coordinates, 
                       hell_state_coordinates=hell_state_coordinates,
                       num_obstacles=len(obstacles_coordinates), 
                       num_treasures=len(treasures_coordinates))
    return env

# Hyperparameters:
learning_rate = 0.1  
gamma = 0.99  
epsilon = 0.1  
epsilon_min = 0.01
epsilon_decay = 0.995
no_episodes = 1000

# Define all hell state coordinates as a tuple within a list
goal_coordinates = (2, 4)
hell_state_coordinates = [(2, 0), (0, 4)]
obstacles_coordinates = [(3, 1), (0, 1),(2,3)]  
treasures_coordinates = [(1, 3),(3,0)] 
random_seed = 42

# Main:
if train:
    # Create an instance of the environment
    env = create_env(goal_coordinates=goal_coordinates,
                     hell_state_coordinates=hell_state_coordinates,
                     obstacles_coordinates=obstacles_coordinates,
                     treasures_coordinates=treasures_coordinates,
                     seed=random_seed)
    
    # Train a Q-learning agent
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)


    # Check if we need to visualize results
if visualize_results:
    # Try to visualize the Q-table
    try:
        visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                          
                          goal_coordinates=goal_coordinates,
                          actions=["Up", "Down", "Right", "Left"],
                          q_values_path="q_table.npy")
    except TypeError as e:
        print(f"Visualization function error: {e}")

    # Load and print the Q-table
    try:
        q_table = np.load("q_table.npy", allow_pickle=True).item()  

        # Print Q-values for each state-action pair
        for state, action_values in q_table.items():
            print(f"State: {state}")
            for action, value in action_values.items():
                print(f"  Action: {action}, Q-value: {value}")
    except Exception as e:
        print(f"Error loading or printing Q-table: {e}")