import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot_running_curve(x, rewards, postscores, penalty, figure_file):
    running_rewards = cal_running_avg(rewards)
    running_postscores = cal_running_avg(postscores)
    running_penalty = cal_running_avg(penalty)
    plt.plot(x, running_rewards, color='blue', label='rewards')
    plt.plot(x, running_postscores, color='green', label='post scores')
    plt.plot(x, running_penalty, color='red', label='penalty')
    plt.legend()
    plt.title('Running average of previous 100 data')
    plt.savefig(figure_file)


def cal_running_avg(x):
    # running_avg = np.zeros(len(x))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(x[max(0, i - 100):(i + 1)])
    # running_avg = np.maximum(running_avg, -30)
    running_avg = np.zeros(len(x)-99)
    for i in range(0, len(x)-99):
        running_avg[i] = np.mean(x[i:i+100])
    return running_avg

