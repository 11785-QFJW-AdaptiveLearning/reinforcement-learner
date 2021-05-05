import numpy as np
import scipy.stats as stats


class BKTStudentSkill:
    def __init__(self, p_L):
        self.mean_pL = p_L
        self.p_L, self.p_T, self.p_S, self.p_G, self.correct = self.init_p_norm(p_L)

    def update(self):
        self.p_L = self.p_L + (1 - self.p_L) * self.p_T
        self.correct = self.p_L * (1 - self.p_S) + (1 - self.p_L) * self.p_G

    def answer(self, is_test):
        if is_test:
            return np.random.binomial(n=1, p=self.correct, size=1)
        else:
            return np.random.binomial(n=1, p=self.p_L, size=1)

    def reset(self):
        self.p_L, self.p_T, self.p_S, self.p_G, self.correct = self.init_p_norm(self.mean_pL)

    def init_p_norm(self, mean_pL):
        low = 0
        high = 1
        p_L = stats.truncnorm.rvs((low - mean_pL) / 0.1, (high - mean_pL) / 0.1, loc=mean_pL, scale=0.1, size=1)
        p_T = stats.truncnorm.rvs((low - 0.8) / 0.2, (high - 0.8) / 0.2, loc=0.8, scale=0.2, size=1)
        p_S = stats.truncnorm.rvs((low - 0.05) / 0.1, (high - 0.05) / 0.1, loc=0.05, scale=0.1, size=1)
        p_G = stats.truncnorm.rvs((low - 0.2) / 0.05, (high - 0.2) / 0.05, loc=0.2, scale=0.05, size=1)
        correct = p_L * (1 - p_S) + (1 - p_L) * p_G
        return p_L, p_T, p_S, p_G, correct


class BKTStudent:
    def __init__(self, num_skills, pretest_per_skill, p_L):
        self.num_skills = num_skills
        self.pretest_per_skill = pretest_per_skill
        self.knowledge_states = [BKTStudentSkill(p_L) for _ in range(num_skills)]

    def answer(self, skill_idx, is_test=1):
        return self.knowledge_states[skill_idx].answer(is_test)

    def updateKnowledge(self, skill_idx):
        self.knowledge_states[skill_idx].update()

    def reset(self):
        for skill in self.knowledge_states:
            skill.reset()

    def takePreTest(self):
        questions = np.repeat(np.array([i for i in range(self.num_skills)]), self.pretest_per_skill)
        scores = [self.answer(question) for question in questions]
        return np.array(scores).reshape((-1,))

    def takePostTest(self):
        questions = np.repeat(np.array([i for i in range(self.num_skills)]), self.pretest_per_skill)
        scores = [self.answer(question) for question in questions]
        return np.array(scores).reshape((-1,))
