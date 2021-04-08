import gym
import numpy as np
from PPO import Agent
from utils import plot_learning_curve
import math
from BKT import BKT

# np.random.seed(11785)


if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env = BKT()
    N = 20
    batch_size = 5
    n_epochs = 1
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 10000

    figure_file = 'rs_nosweet_penalty_pl=0.1_10000.png'

    best_score = -math.inf
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        action_list = []
        reward_list = []
        succ = True
        while not done:
            action, prob, val = agent.choose_action(observation)
            action_list.append(action)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            # print(reward)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if 'stuck' in info:
                print('*Warning: get stuck at action: ', info['stuck'])
                succ = False
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
        print('actions: ', action_list)
        print('skills: ', np.array(action_list)//4)
        print('rewards: ', reward_list)
        print('-' * 50)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
