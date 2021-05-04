import gym
import numpy as np
from PPO import Agent
from utils import plot_learning_curve, plot_running_curve
import math
from BKT import BKT
import datetime

# np.random.seed(11785)


if __name__ == '__main__':
    BKT_param = {'numskill':6, 'activity_per_skill':7, 'pretest_per_skill':3}
    Agent_param = {}
    env = BKT(**BKT_param)
    N = 50
    batch_size = 5
    n_epochs = 3
    alpha = 0.0005
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 5000

    figure_file = 'rs.png'

    best_score = -math.inf
    score_history = []
    post_test_history = []
    penalty_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = []
        action_list = []
        reward_list = []
        succ = True
        info = {}
        while not done:
            action, prob, val = agent.choose_action(observation)
            action_list.append(action)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            agent.updateMask(action, reward)
            # print(reward)
            n_steps += 1
            score.append(reward)
            agent.remember(observation, action, prob, val, reward, done)
            if 'stuck' in info:
                print('*Warning: get stuck at action: ', info['stuck'])
                succ = False
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        agent.resetMask()
        post_test_history.append(info['postscores'])
        score = np.array(score)
        penalty_history.append(np.sum(score[score < 0]))
        score = np.sum(score)
        score_history.append(score)

        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
        print('actions: ', action_list)
        print('skills: ', np.array(action_list) // BKT_param['activity_per_skill'])
        print('rewards: ', reward_list)
        print('-' * 50)
    x = [i + 1 for i in range(len(score_history)-99)]
    result = {
        'post_test_history': post_test_history,
        'penalty_history': penalty_history,
        'score_history': score_history
    }
    bkt_path = '_'.join([str(v) for k, v in BKT_param.items()])
    time = datetime.datetime.now().strftime('%m%d%H%M')
    file_name = f'lpf/baseline_penalty2_bkt_{bkt_path}_{time}'
    np.save(f'{file_name}.npy', result)
    # plot_learning_curve(x, score_history, figure_file)
    plot_running_curve(x, score_history, post_test_history, penalty_history, f'{file_name}.png')
