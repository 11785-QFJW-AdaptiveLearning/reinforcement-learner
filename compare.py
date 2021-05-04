import numpy as np
from utils import *

linear = np.load('lpf/linear_assign_norm2_05041604.npy', allow_pickle=True).item()
rs = np.load('lpf/baseline_penalty2_originstate_norm2_05041600.npy', allow_pickle=True).item()

linear_running_rewards = cal_running_avg(linear['score_history'])
linear_running_postscores = cal_running_avg(linear['post_test_history'])
linear_running_penalty = cal_running_avg(linear['penalty_history'])
rs_running_rewards = cal_running_avg(rs['score_history'])
rs_running_postscores = cal_running_avg(rs['post_test_history'])
rs_running_penalty = cal_running_avg(rs['penalty_history'])
x = [i + 1 for i in range(len(linear_running_rewards))]
plt.plot(x, linear_running_rewards, color='blue', label='linear rewards')
plt.plot(x, linear_running_postscores, color='green', label='linear post scores')
plt.plot(x, linear_running_penalty, color='red', label='linear penalty')
plt.plot(x, rs_running_rewards, color='lightblue', label='rs rewards')
plt.plot(x, rs_running_postscores, color='lightgreen', label='rs post scores')
plt.plot(x, rs_running_penalty, color='lightcoral', label='rs penalty')
plt.legend()
plt.title('Running average of previous 100 data')
plt.savefig('lpf/compare.png')
