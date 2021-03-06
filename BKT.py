import numpy as np
from gym import spaces, Env
from BKTStudent import BKTStudent


class BKT(Env):
    """
    Description:
        A student's response with ASSIGNED activities. The student performs under Bayesian Knowledge Tracing
        model.

    Observation:
        Type: Box(4*3)
        Num     Observation               Min                     Max
        0-3     Pre-Score                  0                       1
        3-6     Complete Status            0                       1
        6-9     Post-Score                 0                       1

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Assign Activity 0
        1     Assign Activity 1
        2     Assign Acitivity 2
        3     Assign Post-Test


    Reward:
        Reward is 1 for every step taken. At terminal step, it follows by formula in Paper

    Starting State:
        Pre-Scores for each activity are assigned a uniform random value in either 0 or 1

    Episode Termination:
        Post-Test is completed

    --------------------------------------------------------
    Setup:

    12 Activities
    Questions in each activitiy

    3 Skills
    6 Pre/Post Test Questions

    0 1 2 3    skill 0
    4 5 6 7    skill 1
    8 9 10 11  skill 2
    --------------------------------------------------------
    """

    def __init__(self, numskill, activity_per_skill, pretest_per_skill,
                 penalty, learned_discount, learned_penalty, learned_sweet, p_L):
        range = numskill * pretest_per_skill + numskill * activity_per_skill + numskill * activity_per_skill
        LOW = np.zeros(range)  # 6 pre-tests, 12 activities, 12 activities' grads
        HIGH = np.ones(range)
        self.student = BKTStudent(num_skills=numskill, pretest_per_skill=pretest_per_skill, p_L=p_L)
        self.action_space = spaces.Discrete(numskill * activity_per_skill + 1)
        self.observation_space = spaces.Box(LOW, HIGH, dtype=np.double)
        self.assigned = []
        self.assigned_count = []
        self.state = np.zeros(self.observation_space.shape, dtype=int)
        self.penalty = penalty
        self.learned_discount = learned_discount
        self.learned_penalty = learned_penalty
        self.learned_sweet = learned_sweet
        self.numskill = numskill
        self.activity_per_skill = activity_per_skill
        self.pre_test_cnt = numskill * pretest_per_skill
        self.pretest_per_skill = pretest_per_skill

    def step(self, action):
        action = int(action)
        done = False
        skill = None
        reward = 0.
        info = {}

        # check which skill this action belongs
        max_action = self.numskill * self.activity_per_skill
        if action < max_action:
            skill = action // self.activity_per_skill

        # if the system assigns an activity to the student
        if action != max_action and skill is not None:
            # check if assigned
            if action not in self.assigned:
                # record the assigned data
                self.assigned.append(action)
                self.assigned_count.append(1)
                # print("assigned")
                # take post activity practice, record the score, update knowledge state
                self.student.updateKnowledge(skill)
                # activity type
                activity_is_test = 1 if action % self.activity_per_skill == self.activity_per_skill - 1 else 0
                activity_score = self.student.answer(skill, activity_is_test)[0]
                reward = self.learned_discount ** (
                            np.mean(self.state[skill * self.pretest_per_skill:(skill + 1) * self.pretest_per_skill])
                            + np.sum(self.state[self.pre_test_cnt + max_action + skill * self.activity_per_skill:
                                                self.pre_test_cnt + max_action + (skill + 1) * self.activity_per_skill]))

                self.state[self.pre_test_cnt + action] = 1
                self.state[self.pre_test_cnt + max_action + action] = activity_score
                # reward = 1.
                # print(reward)
            else:  # already assigned
                raise Warning("already assigned")
                print('======================= already assigned!')
                idx = self.assigned.index(action)
                # check if learned
                if self.state[self.pre_test_cnt + max_action + action] == 0:
                    # not learned, reduced reward
                    reward = self.learned_discount ** self.assigned_count[idx]
                else:
                    if self.assigned_count[idx] >= 4:
                        info['stuck'] = action
                    # learned, penalty reward
                    reward = -self.learned_penalty ** self.assigned_count[idx]
                self.assigned_count[idx] += 1

        # if the system assigns post test to the student
        elif action == max_action:
            # print("post")
            done = True
            postsocres = self.student.takePostTest()
            info['postscores'] = np.sum(postsocres)
            prescores = self.state[0:self.pre_test_cnt]
            # reward = 1.0 + np.sum(postsocres - prescores) - (1 + self.penalty) * len(self.assigned)
            # reward = 1.0 + self.learned_sweet * np.sum(np.maximum(postsocres - prescores, 0)) - (1 + self.penalty) * len(self.assigned)
            if len(self.assigned) > 0:
                reward = 1.0 \
                         + self.learned_sweet * np.sum(np.maximum(postsocres - prescores, 0)) \
                         - (1 + self.penalty) * len(self.assigned)
            else:
                raise Warning("only post test")
                print('^^^^^^^^^^^^^^^^^^^^^^^ only post test!')
                reward = -(max_action) + 2 * np.sum(np.minimum(postsocres - prescores, 0))
            print('pre-test:\t', prescores)
            print('post-test:\t', postsocres)

        return self.state, reward, done, info

    def reset(self):
        self.student.reset()
        self.state = np.zeros(self.observation_space.shape, dtype=int)
        self.assigned = []
        self.assigned_count = []

        # do pretest after reset & before assign activities
        presocres = self.student.takePreTest()
        self.state[0:len(presocres)] = presocres
        return self.state

# if __name__ == "__main__":
#     BKT_param = {'numskill': 6, 'activity_per_skill': 7, 'pretest_per_skill': 3}
#     env = BKT(**BKT_param)
#     print(env.action_space)
#     print(env.observation_space)
