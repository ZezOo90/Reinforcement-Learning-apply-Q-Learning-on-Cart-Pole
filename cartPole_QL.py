import gym
import numpy as np
import time
import matplotlib.pyplot as plt


class Q_Learning:
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
            numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))
        self.sumRewardsEpisode = []

    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def select_action(self, state, index):
        if np.random.random() < self.epsilon or index < 100:
            return np.random.choice(self.actionNumber)
        return np.argmax(self.Qmatrix[self.returnIndexState(state)])

    def simulate_episodes(self):
        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode = []
            state, _ = self.env.reset()
            terminalState = False

            while not terminalState:
                action = self.select_action(state, indexEpisode)
                next_state, reward, terminalState, _, _ = self.env.step(action)
                rewardsEpisode.append(reward)
                q_next_max = np.max(self.Qmatrix[self.returnIndexState(next_state)])
                q_target = reward + self.gamma * q_next_max
                self.Qmatrix[self.returnIndexState(state), action] += self.alpha * (
                            q_target - self.Qmatrix[self.returnIndexState(state), action])
                state = next_state

            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
            if (indexEpisode + 1) % 100 == 0:
                print(f"Episode {indexEpisode + 1}: Total reward = {self.sumRewardsEpisode[-1]}")

    def simulate_learned_strategy(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        state, _ = env1.reset()
        env1.render()
        obtained_rewards = []
        for _ in range(1000):
            action = np.argmax(self.Qmatrix[self.returnIndexState(state)])
            state, reward, done, _, _ = env1.step(action)
            obtained_rewards.append(reward)
            if done:
                time.sleep(1)  # Add a delay after the episode ends
                break
        return obtained_rewards


# Parameters
env = gym.make('CartPole-v1')
alpha = 0.1
gamma = 0.95
epsilon = 0.2
number_episodes = 500

lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
upper_bounds[1] = 3
upper_bounds[3] = 2
lower_bounds[1] = -3
lower_bounds[3] = -2

number_bins = [10, 10, 10, 10]

# Q-Learning
agent = Q_Learning(env, alpha, gamma, epsilon, number_episodes, number_bins, lower_bounds, upper_bounds)
agent.simulate_episodes()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(agent.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()

# Simulate learned strategy
obtained_rewards = agent.simulate_learned_strategy()



