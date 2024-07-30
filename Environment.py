#Importing
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Define the class TreasureHunt inherited from gym.Env
class TreasureHunt(gym.Env):
    def __init__(self, grid_size=5, seed=None, goal_coordinates=(2, 4), 
                 hell_state_coordinates=[], num_obstacles=3, num_treasures=2) -> None:
        super(TreasureHunt, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal_state = np.array(goal_coordinates)
        self.hell_states = [np.array(coord) for coord in hell_state_coordinates]
        self.done = False
        self.num_obstacles = num_obstacles
        self.num_treasures = num_treasures

        # Making the environment static
        if seed is not None:
            np.random.seed(seed)
            
        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(
            low=0, high=np.array([grid_size-1, grid_size-1]), shape=(2,), dtype=np.int32)

        self.agent_state = np.array([1, 1])
        self.obstacles = self._generate_obstacles()
        self.treasures = self._generate_treasures()

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

        # Load images
        self.theme = mpimg.imread('./images/theme.jpg')
        self.agent_img = mpimg.imread('./images/agent.png')
        self.treasure_img = mpimg.imread('./images/treasure.png')
        self.obstacle_img = mpimg.imread('./images/obstelcale.png')
        self.hell_img = mpimg.imread('./images/hell_state.png')
        self.goal_img = mpimg.imread('./images/goal.png')

    # Define _generate_positions method
    def _generate_positions(self, num_positions, exclude_positions):
        positions = []
        for _ in range(num_positions):
            position = np.random.randint(0, self.grid_size, size=2)
            while any(np.array_equal(position, ep) 
                for ep in exclude_positions) or any(np.array_equal(position, p) 
                    for p in positions):
                    position = np.random.randint(0, self.grid_size, size=2)
            positions.append(position)
        return positions

    # Define _generate_obstacles method
    def _generate_obstacles(self):
        return self._generate_positions(self.num_obstacles, [self.agent_state, self.goal_state])

    # Define _generate_treasures method
    def _generate_treasures(self):
        occupied_positions = [self.agent_state, self.goal_state] + self.obstacles
        return self._generate_positions(self.num_treasures, occupied_positions)

    # Define add_hell_states method
    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    # Define the reset method
    def reset(self):
         # Generate a random starting position for the agent
        self.agent_state = np.random.randint(0, self.grid_size, size=2)
        while any(np.array_equal(self.agent_state, e) 
                  for e in [self.goal_state] + self.obstacles + self.treasures + self.hell_states):
            self.agent_state = np.random.randint(0, self.grid_size, size=2)
        
        self.state = self.agent_state
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal_state[0])**2 +
            (self.state[1] - self.goal_state[1])**2
        )

        return self.state, self.info

    # Define the step method
    def step(self, action):
        # Store the current state for comparison later
        old_state = np.copy(self.agent_state)

        if action == 0 and self.agent_state[1] < self.grid_size - 1:  # up
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:  # right
            self.agent_state[0] += 1

        # Initialize the reward to -0.1
        reward = -0.1
        done=False
        if np.array_equal(self.agent_state, self.goal_state):
            reward += 10
            done=True
        
        if np.any([np.array_equal(self.agent_state, t) 
                   for t in self.obstacles]):
            reward -= 1
            done=True
        
        if np.any([np.array_equal(self.agent_state, h) 
                   for h in self.hell_states]):
            reward -= 10
            done=True

        if np.any([np.array_equal(self.agent_state, u) 
                   for u in self.treasures]):
            reward += 1
            done=True

        # Calculating Euclidean Distance
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}
        return self.agent_state, reward, done, info

    # Define the render method
    def render(self):
        self.ax.clear()
        # Adding a theme
        self.ax.imshow(self.theme, extent=[-1, self.grid_size, -1, self.grid_size])
        
        # Render agent
        img = OffsetImage(self.agent_img, zoom=0.05)
        ab = AnnotationBbox(img, (self.agent_state[0], self.agent_state[1]), frameon=False)
        self.ax.add_artist(ab)

        # Render treasures
        for treasure in self.treasures:
            img = OffsetImage(self.treasure_img, zoom=0.07)
            ab = AnnotationBbox(img, (treasure[0], treasure[1]), frameon=False)
            self.ax.add_artist(ab)

        # Render obstacles
        for obstacle in self.obstacles:
            img = OffsetImage(self.obstacle_img, zoom=0.07)
            ab = AnnotationBbox(img, (obstacle[0], obstacle[1]), frameon=False)
            self.ax.add_artist(ab)

        # Render hell states
        for hell_state in self.hell_states:
            img = OffsetImage(self.hell_img, zoom=0.07)
            ab = AnnotationBbox(img, (hell_state[0], hell_state[1]), frameon=False)
            self.ax.add_artist(ab)
            
        # Render goal
        img = OffsetImage(self.goal_img, zoom=0.05)
        ab = AnnotationBbox(img, (self.goal_state[0], self.goal_state[1]), frameon=False)
        self.ax.add_artist(ab)

        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        plt.pause(0.1)

    # Define the close method
    def close(self):
        plt.close()

#Creating an instance of the TreasureHunt
if __name__ == "__main__":
    env = TreasureHunt(seed=42)
    state = env.reset()

    for step in range(500):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        print(f"Step: {step+1}, State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Goal reached! Mission accomplished!")
            break

    # close the environment
    env.close()