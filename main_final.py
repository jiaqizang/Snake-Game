import math
import pygame
import random
import numpy as np
from datetime import datetime
from collections import deque
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define a snake game class
class SnakeGame:
    def __init__(self, width=640, height=480, grid_size=40):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.reset()

    # Function to reset/restart the game
    def reset(self):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont('Arial', 30)
        self.score = 0

        # Initialize snake, the initial position is fixed for all games
        self.snake = deque()
        self.snake.append((self.width / 2, self.height / 2))

        # Initialize food
        self.food = self.get_random_pos()

        # Return the initial state
        return self.get_state(0)

    # Function to assign a random position to the food
    def get_random_pos(self):
        x = random.randint(0, self.width - self.grid_size)
        y = random.randint(0, self.height - self.grid_size)
        return (x // self.grid_size * self.grid_size, y // self.grid_size * self.grid_size)

    # Function to take a step in the game
    def step(self, action):
        # Action space: 0 - UP, 1 - RIGHT, 2 - DOWN, 3 - LEFT

        # Initialize the reward
        reward = 0

        # Get the current position of the snake head
        x, y = self.snake[0]

        # Update the position of the snake head
        if action == 0:
            y -= self.grid_size
        elif action == 1:
            x += self.grid_size
        elif action == 2:
            y += self.grid_size
        elif action == 3:
            x -= self.grid_size

        # Check if the snake is dead, if dead, reward is -200
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake:
            reward = -100
            return False, self.score, reward

        # Update the position of the snake
        self.snake.appendleft((x, y))

        # Check if the snake eats the food
        if (x, y) == self.food:
            reward = 10
            self.score += 1
            self.food = self.get_random_pos()
            # Check if the new food is on the snake, if so, assign a new position to the food
            while self.food in self.snake:
                self.food = self.get_random_pos()

        # else if the snake gets closer to the food, if so, reward is 1 and remove the snake tail
        elif (x - self.food[0]) ** 2 + (y - self.food[1]) ** 2 < (self.snake[1][0] - self.food[0]) ** 2 + (self.snake[1][1] - self.food[1]) ** 2:
            reward = 1
            self.snake.pop()

        # else if the snake gets farther to the food, if so, reward is -1 and remove the snake tail
        else:
            reward = -1
            self.snake.pop()

        return True, self.score, reward

    # Function to draw the game in a UI window
    def draw(self, action):
        # Draw the background
        self.screen.fill((0, 0, 0))

        # Draw the snake body in rectangle shape
        color = 255
        for x, y in self.snake:
            # Draw the snake head in green color
            if (x, y) == self.snake[0]:
                pygame.draw.rect(self.screen, (0, 255, 0), (x, y, self.grid_size, self.grid_size))
            else:
                color = color - 5 if color - 5 > 0 else 5
                pygame.draw.rect(self.screen, (color, color, color), (x, y, self.grid_size, self.grid_size))

        # Draw the food in red color
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.grid_size, self.grid_size))

        # Display the score
        score_surface = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 10))

        # Update the display
        pygame.display.flip()

    # Function to run the game by one step, display the game, and return the state
    def run(self, action):
        running, score, reward = self.step(action)
        state = self.get_state(action)
        self.draw(action)
        self.clock.tick(1000)
        return running, score, reward, state

    # Function to get the state of the game
    def get_state(self, action):
        # Get the position of the snake head
        x, y = self.snake[0]

        # Get the position of the food
        food_x, food_y = self.food

        # Initialize 4 states of binary values
        snake_dir_up = 0
        snake_dir_right = 0
        snake_dir_down = 0
        snake_dir_left = 0

        if action == 0:
            snake_dir_up = 1
        elif action == 1:
            snake_dir_right = 1
        elif action == 2:
            snake_dir_down = 1
        elif action == 3:
            snake_dir_left = 1

        # Check if the food is up to the snake head
        food_up = 1 if food_y < y else 0

        # Check if the food is right to the snake head
        food_right = 1 if food_x > x else 0

        # Check if the food is down to the snake head
        food_down = 1 if food_y > y else 0

        # Check if the food is left to the snake head
        food_left = 1 if food_x < x else 0

        # Check if wall or snake body is up to the snake head
        obstacle_up = 1 if y == 0 or (x, y - self.grid_size) in self.snake else 0

        # Check if wall or snake body is right to the snake head
        obstacle_right = 1 if x == self.width - self.grid_size or (x + self.grid_size, y) in self.snake else 0

        # Check if wall or snake body is down to the snake head
        obstacle_down = 1 if y == self.height - self.grid_size or (x, y + self.grid_size) in self.snake else 0

        # Check if wall or snake body is left to the snake head
        obstacle_left = 1 if x == 0 or (x - self.grid_size, y) in self.snake else 0

        # Check if the snake is moving towards the body
        body_up = 1 if (snake_dir_up and (x, y - self.grid_size) in self.snake) else 0
        body_right = 1 if (snake_dir_right and (x + self.grid_size, y) in self.snake) else 0
        body_down = 1 if (snake_dir_down and (x, y + self.grid_size) in self.snake) else 0
        body_left = 1 if (snake_dir_left and (x - self.grid_size, y) in self.snake) else 0
        enclose = 1 if body_up or body_right or body_down or body_left else 0

        # Return the state
        return np.array([snake_dir_up, snake_dir_right, snake_dir_down, snake_dir_left, food_up, food_right, food_down, food_left, obstacle_up, obstacle_right, obstacle_down, obstacle_left, enclose])

# Define a DQN agent
def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # Input Layer of state size 12 and Hidden Layer with 128 nodes
    X = Dense(128, input_shape=input_shape, activation="relu", )(X_input)

    # Hidden layer with 128 nodes
    X = Dense(128, activation="relu")(X)

    # Hidden layer with 128 nodes
    X = Dense(128, activation="relu")(X)

    # Output Layer with # of actions: 4 nodes (up, right, down, left)
    X = Dense(action_space, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X, name='SnakeGame_DQN_model')
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.00025), metrics=["accuracy"])

    # model.summary()
    return model

# Define a DQN agent
class DQNAgent:
    def __init__(self):
        self.env = SnakeGame()
        self.state_size = 13
        self.action_size = 4
        self.EPISODES = 50
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 500
        self.train_start = 500

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    # Function to remember the state, the action, the reward, and the next state
    def remember(self, state, action, reward, next_state, running):
        self.memory.append((state, action, reward, next_state, running))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # Function to pick an action given a state, note that the snake is not allowed to go back
    def act(self, state):
        while True:
            # exploration
            if np.random.rand() <= self.epsilon:
                action_back = random.randrange(self.action_size)
                if action_back == 0 and state[0][2] == 1:
                    continue
                elif action_back == 1 and state[0][3] == 1:
                    continue
                elif action_back == 2 and state[0][0] == 1:
                    continue
                elif action_back == 3 and state[0][1] == 1:
                    continue
                else:
                    return action_back
            # exploitation
            else:
                action_back = np.argmax(self.model.predict(state, verbose=0)[0])
                if action_back == 0 and state[0][2] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 0:
                        action_back = random.randrange(self.action_size)
                elif action_back == 1 and state[0][3] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 1:
                        action_back = random.randrange(self.action_size)
                elif action_back == 2 and state[0][0] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 2:
                        action_back = random.randrange(self.action_size)
                elif action_back == 3 and state[0][1] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 3:
                        action_back = random.randrange(self.action_size)

                return action_back


    # Function to replay the memory and train the network
    def replay(self):
        if len(self.memory) < self.train_start:
            return

        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, running = [], [], []

        # assign data into state, next_state, action, reward and running from minibatch
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            running.append(minibatch[i][4])

        # Compute the value function of current and next state
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            if not running[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches where target is the value function
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # Function to load the model for testing purpose
    def load(self, name):
        self.model = load_model(name)

    # Function to save the model for testing purpose
    def save(self, name):
        self.model.save(name)

    # Function to train the model
    def training(self):
        total_episodes = []
        total_rewards = []
        total_scores = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            running = True
            i = 0
            while running:
                action = self.act(state)
                running, score, reward, next_state = self.env.run(action)

                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, reward, next_state, running)
                state = next_state

                i += reward
                if not running:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("episode: {}/{}, score: {}, total reward: {}, e: {:.2}, time: {}".format(e + 1, self.EPISODES, score, i, self.epsilon,
                                                                                 timestampStr))
                    total_episodes.append(e + 1)
                    total_rewards.append(i)
                    total_scores.append(score)

                self.replay()

        # Save the trained model
        print("Saving trained model as SnakeGame-dqn-training.h5")
        self.save("SnakeGame-dqn-training.h5")

        return total_episodes, total_rewards, total_scores

    # test function if you want to test the learned model
    def test(self):
        self.load("SnakeGame-dqn-training.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            running = True
            i = 0
            while running:
                action = np.argmax(self.model.predict(state, verbose=0))
                running, score, reward, next_state = self.env.run(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += reward

                if not running:
                    print("episode: {}/{}, reward: {}".format(e + 1, self.EPISODES, i))
                    break

# Define a Double DQN agent
class DDQNAgent:
    def __init__(self):
        self.env = SnakeGame()
        self.state_size = 13
        self.action_size = 4
        self.EPISODES = 50
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 500
        self.train_start = 500
        self.TARGET_UPDATE_FREQUENCY = 100

        # create main model and target model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
        # Initialize target model
        self.update_target_model()

    # Function to update the target model to be same with model, after some time interval
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Function to remember the state, the action, the reward, and the next state
    def remember(self, state, action, reward, next_state, running):
        self.memory.append((state, action, reward, next_state, running))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # Function to pick an action given a state, note that the snake is not allowed to go back
    def act(self, state):
        while True:
            # exploration
            if np.random.rand() <= self.epsilon:
                action_back = random.randrange(self.action_size)
                if action_back == 0 and state[0][2] == 1:
                    continue
                elif action_back == 1 and state[0][3] == 1:
                    continue
                elif action_back == 2 and state[0][0] == 1:
                    continue
                elif action_back == 3 and state[0][1] == 1:
                    continue
                else:
                    return action_back
            # exploitation
            else:
                action_back = np.argmax(self.model.predict(state, verbose=0)[0])
                if action_back == 0 and state[0][2] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 0:
                        action_back = random.randrange(self.action_size)
                elif action_back == 1 and state[0][3] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 1:
                        action_back = random.randrange(self.action_size)
                elif action_back == 2 and state[0][0] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 2:
                        action_back = random.randrange(self.action_size)
                elif action_back == 3 and state[0][1] == 1:
                    action_back = random.randrange(self.action_size)
                    while action_back == 3:
                        action_back = random.randrange(self.action_size)

                return action_back


    # Function to replay the memory and train the network
    def replay(self):
        if len(self.memory) < self.train_start:
            return

        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, running = [], [], []

        # assign data into state, next_state, action, reward and running from minibatch
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            running.append(minibatch[i][4])

        # Compute the value function of current state using the primary network
        target = self.model.predict(state, verbose=0)

        # For the next state, use the policy network to select the best action and the target network to evaluate this action
        action_values_next_state = self.model.predict(next_state, verbose=0)
        best_action_next_state = np.argmax(action_values_next_state, axis=1)
        target_next_model = self.target_model.predict(next_state, verbose=0)
        Q_values_next_state = target_next_model[range(self.batch_size), best_action_next_state]

        for i in range(self.batch_size):
            if not running[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * Q_values_next_state[i]

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # Function to load the model for testing purpose
    def load(self, name):
        self.model = load_model(name)

    # Function to save the model for testing purpose
    def save(self, name):
        self.model.save(name)

    # Function to train the model
    def training(self):
        total_episodes = []
        total_rewards = []
        total_scores = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            running = True
            i = 0
            while running:
                action = self.act(state)
                running, score, reward, next_state = self.env.run(action)

                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, reward, next_state, running)
                state = next_state

                i += reward
                if not running:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print("episode: {}/{}, score: {}, total reward: {}, e: {:.2}, time: {}".format(e + 1, self.EPISODES, score, i, self.epsilon,
                                                                                 timestampStr))

                    total_episodes.append(e + 1)
                    total_rewards.append(i)
                    total_scores.append(score)

                self.replay()

            # Update target network every TARGET_UPDATE_FREQUENCY episodes
            if e % self.TARGET_UPDATE_FREQUENCY == 0:
                self.update_target_model()

        return total_episodes, total_rewards, total_scores

    # test function if you want to test the learned model
    def test(self):
        self.load("SnakeGame-dqn-training.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            running = True
            i = 0
            while running:
                action = np.argmax(self.model.predict(state, verbose=0))
                running, score, reward, next_state = self.env.run(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += reward

                if not running:
                    print("episode: {}/{}, reward: {}".format(e + 1, self.EPISODES, i))
                    break


if __name__ == "__main__":
    agent_DQN = DQNAgent()
    episodes_DQN, rewards_DQN, scores_DQN = agent_DQN.training()

    agent_DDQN = DDQNAgent()
    episodes_DDQN, rewards_DDQN, scores_DDQN = agent_DDQN.training()

    # plot the rewards of these two agents in one figure
    plt.figure(figsize=(12, 8))
    plt.plot(episodes_DQN, rewards_DQN, label='DQN', color='red')
    plt.plot(episodes_DDQN, rewards_DDQN, label='DDQN', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards per Episode')
    plt.legend()
    plt.show()

    # plot the scores of these two agents in one figure
    plt.figure(figsize=(12, 8))
    plt.plot(episodes_DQN, scores_DQN, label='DQN', color='red')
    plt.plot(episodes_DDQN, scores_DDQN, label='DDQN', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Total Scores per Episode')
    plt.legend()
    plt.show()

    agent_DQN.test()