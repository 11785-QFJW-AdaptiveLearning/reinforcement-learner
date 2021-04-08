import numpy as np
from gym import spaces, Env
from BKTStudent import BKTStudent, BKTStudentSkill

LOW = np.zeros(6+12+6)
HIGH = np.ones(6+12+6)

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

    def __init__(self):
        self.student = BKTStudent(num_skills = 12)
        self.action_space = spaces.Discrete(12+1)
        self.observation_space = spaces.Box(LOW, HIGH, dtype=np.double)
        self.assigned = []
        self.state = np.zeros(self.observation_space.shape)
        self.penalty = 0.1


    def step(self, action):
        action = int(action)
        done = False
        skill = None
        reward = 0.

        # if not taken pre test yet, take the test and update the state
        if len(self.assigned) == 0:
            # print("start")
            presocres = self.student.takePreTest()
            self.state[0:len(presocres)] = presocres

        # check which skill this action belongs
        if action <= 3:
            skill = 0
        elif action <= 7:
            skill = 1
        elif action <= 11:
            skill = 2

        # if the system assigns an activity to the student
        if action != 12 and skill is not None:
            # record the assigned data
            if action not in self.assigned:
                self.assigned.append(action)
                # print("assigned")
                # take post activity practice, record the score, update knowledge state
                activity_score = self.student.answer(skill)
                self.student.updateKnowledge(activity_score, skill)
                self.state[6+action] = 1.
                reward = 1.
                # print(reward)

        # if the system assigns post test to the student
        if action == 12:
            # print("post")
            done = True
            postsocres = self.student.takePostTest()
            self.state[18:18+len(postsocres)] = postsocres
            prescores = self.state[0:6]
            reward = 1.0 + np.sum(postsocres - prescores) - (1 + self.penalty) * len(self.assigned)

        return self.state, reward, done, {}

    def reset(self):
        self.student.reset()
        self.state = np.zeros(self.observation_space.shape)
        self.assigned = []
        return self.state




env = BKT()
env.action_space
env.observation_space


























