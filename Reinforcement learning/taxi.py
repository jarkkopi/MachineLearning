import numpy as np
import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
print(env.render())

alpha = 0.05    # lr
gamma = 0.9    # discount fac
epsilon = 0.1  # exploration rate
episodes = 10000  # episodes

state_space = env.observation_space.n
action_space = env.action_space.n

Q_table = np.zeros((state_space, action_space))

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state

total_rewards = []
total_actions = []

for episode in range(10): 
    state, _ = env.reset()
    done = False
    total_reward = 0
    actions = 0

    while not done:
        action = np.argmax(Q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        actions += 1
        state = next_state
    
    total_rewards.append(total_reward)
    total_actions.append(actions)

print(f"Average total reward: {np.mean(total_rewards)}")
print(f"Average num of actions: {np.mean(total_actions)}")
