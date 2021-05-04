import numpy as np
import scipy.stats as stats
from scipy.special import expit

class IRTStudentSkill:
    def __init__(self):
        """
        a : Item discrimination, determines how sharp the rise in difficulty
            is, and controls the maximum slope of the probability curve,
            which is given by a / 4.
        b : Item difficulty, controls the `theta` value which yields
            the maximal slope, which is given by ``-b / a``
        theta: Student ability, similar to P_L in BKT
        c : The pseudo-guessing probability, the minimal chance of success
            when the ability decreases to negative infinity. similar to P_G in BKT
        d : One minus the inattention probability, or alternatively the
            maximal chance of success when the ability increases to positive
            infinity. similar to P_S in BKT

        """
        self.theta, self.p_T, self.a, self.b, self.d, self.c = self.init_p_norm()
        self.correct = self.cal_correct()

    def update(self):
        self.theta = self.theta + (1 - self.theta) * self.p_T
        self.correct = self.cal_correct()

    def answer(self, is_test):
        if is_test:
            return np.random.binomial(n=1, p=self.correct, size=1)
        else:
            return np.random.binomial(n=1, p=self.theta, size=1)

    def reset(self):
        self.theta, self.p_T, self.a, self.b, self.d, self.c = self.init_p_norm()
        self.correct = self.cal_correct()

    def cal_correct(self):
        return self.c + (self.d - self.c) * expit(self.a * self.theta + self.b)

    def init_p_norm(self):
        low = 0
        high = 1
        theta = stats.truncnorm.rvs((low - 0.5) / 0.1, (high - 0.5) / 0.1, loc=0.5, scale=0.1, size=1)
        p_T = stats.truncnorm.rvs((low - 0.8) / 0.2, (high - 0.8) / 0.2, loc=0.8, scale=0.2, size=1)
        a = stats.truncnorm.rvs((low - 0.05) / 0.1, (high - 0.05) / 0.1, loc=0.05, scale=0.1, size=1)
        b = stats.truncnorm.rvs((low - 0.2) / 0.05, (high - 0.2) / 0.05, loc=0.2, scale=0.05, size=1)
        d = stats.truncnorm.rvs((low - 0.05) / 0.1, (high - 0.05) / 0.1, loc=0.05, scale=0.1, size=1)
        c = stats.truncnorm.rvs((low - 0.2) / 0.05, (high - 0.2) / 0.05, loc=0.2, scale=0.05, size=1)
        return theta, p_T, a, b, d, c


class IRTStudent:
    def __init__(self, num_skills):
        self.num_skills = num_skills
        self.knowledge_states = [IRTStudentSkill() for _ in range(num_skills)]

    def answer(self, skill_idx, is_test=1):
        return self.knowledge_states[skill_idx].answer(is_test)

    def updateKnowledge(self, skill_idx):
        self.knowledge_states[skill_idx].update()

    def reset(self):
        for skill in self.knowledge_states:
            skill.reset()

    def takePreTest(self):
        questions = np.repeat(np.array([0, 1, 2]), 2)
        scores = [self.answer(question) for question in questions]
        return np.array(scores).reshape((-1,))

    def takePostTest(self):
        questions = np.repeat(np.array([0, 1, 2]), 2)
        scores = [self.answer(question) for question in questions]
        return np.array(scores).reshape((-1,))

# class IRTSkillItem:
#     def __init__(self, num_skill_items):


s = IRTStudent(num_skills=3)
s.takePreTest()
