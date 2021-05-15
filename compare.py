import numpy as np
from utils import *

linear = np.load('result_plot/linear_assign_bkt_5_4_5_0.5_0.1_0.8_1.5_1_05060246.npy', allow_pickle=True).item()
rs = np.load('result_plot/baseline_bkt_5_4_5_0.5_0.1_0.8_1.5_1_actor_65@256@256_critic_65@256@256_05060257.npy',
             allow_pickle=True).item()

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
plt.savefig('result_plot/compare_05060246&05060257.png')
plt.show()

linear_rewards_mean = np.mean(linear['score_history'][-1000:])
linear_postscores_mean = np.mean(linear['post_test_history'][-1000:])
linear_penalty_mean = np.mean(linear['penalty_history'][-1000:])
rs_rewards_mean = np.mean(rs['score_history'][-1000:])
rs_postscores_mean = np.mean(rs['post_test_history'][-1000:])
rs_penalty_mean = np.mean(rs['penalty_history'][-1000:])
print("linear_rewards_mean: ", linear_rewards_mean)
print("linear_postscores_mean: ", linear_postscores_mean)
print("linear_penalty_mean: ", linear_penalty_mean)
print("rs_rewards_mean: ", rs_rewards_mean)
print("rs_postscores_mean: ", rs_postscores_mean)
print("rs_penalty_mean: ", rs_penalty_mean)
