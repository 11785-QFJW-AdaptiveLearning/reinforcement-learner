import gym
import numpy as np
from utils import plot_learning_curve, plot_running_curve
import math
from BKT import BKT
import datetime

# np.random.seed(11785)


if __name__ == '__main__':
    BKT_param = {'numskill': 3, 'activity_per_skill': 4, 'pretest_per_skill': 2,
                 'penalty': 0.1, 'learned_discount': 0.5, 'learned_penalty': 1.5, 'learned_sweet': 1}
    env = BKT(**BKT_param)
    N = 50
    n_games = 3000

    score_history = []
    post_test_history = []
    penalty_history = []

    avg_score = 0
    n_steps = 0

    actions = np.arange(BKT_param['numskill']*BKT_param['activity_per_skill']+1)

    for i in range(n_games):
        observation = env.reset()
        score = []
        action_list = []
        reward_list = []
        info = {}
        for action in actions:
            action_list.append(action)
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            # print(reward)
            n_steps += 1
            score.append(reward)
            observation = observation_
        post_test_history.append(info['postscores'])
        score = np.array(score)
        penalty_history.append(np.sum(score[score < 0]))
        score = np.sum(score)
        score_history.append(score)

        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps)
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
    file_name = f'result_plot/linear_assign_bkt_{bkt_path}_{time}'
    np.save(f'{file_name}.npy', result)
    # plot_learning_curve(x, score_history, figure_file)
    plot_running_curve(x, score_history, post_test_history, penalty_history, f'{file_name}.png')
