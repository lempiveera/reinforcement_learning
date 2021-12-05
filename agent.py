# -*- coding: utf-8 -*-
from numpy.core.records import record
import torch
import random
import numpy
from collections import deque  # datastructure where we store memory
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotting import plot

MAX_MEMORY = 100000 # can store this much items
BATCH_SIZE = 1000
LR = 0.001  # LR = learning rate


class Agent:

    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # parameter to control randomness
        self.gamma = 0.85  # discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # if max memory is exceeted it will automatically remove items from the left with popleft() function
        self.model = Linear_QNet(11, 256, 3) # input, hidden and output size, inputstate has 11 values, output [0,0,1] = the dircetion/turn
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        # 4 points created around the snake head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        #check direction of snake, one of these is 1(true) others 0(false)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l, 
            dir_r, 
            dir_u, 
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
            ]

        return numpy.array(state, dtype=int) # convert list to a numpy array, converts true or false to 1 or 0

    def remember(self, state, action, reward, next_state, done): # done = game over
        self.memory.append((state, action, reward, next_state, done)) # popleft if max mem is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples (whatever that means)
        else: 
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)   # puts states together, actions together etc.
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done): # one step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves : tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # snake decides does it do a random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) #converting to a tensor
            prediction = self.model(state0) # model predicts
            move = torch.argmax(prediction).item() # item() converts tensor to a number
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []  # average scores
    total_score = 0
    record = 0
    agent = Agent() # creating the agent
    game = SnakeGameAI() # creating the game
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move based on state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory of agent (one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done) #final move = action

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # if true (game over), train long memory(also called replay memory), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            print('score', score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            print('mean score', mean_score)
            plot(plot_scores, plot_mean_scores) # plot_scores, plot_mean_scores
            


if __name__ == '__main__':
    train()
