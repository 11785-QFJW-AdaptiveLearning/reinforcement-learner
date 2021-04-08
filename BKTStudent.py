import numpy as np

class BKTStudentSkill:
    def __init__(self):
        np.random.seed(100)
        self.p_L = np.random.uniform(low = 0, high = 1, size = 1)
        self.p_T = np.random.uniform(low= 0, high = 1, size = 1)
        self.p_S = np.random.uniform(low = 0, high = 1, size = 1)
        self.p_G = np.random.uniform(low = 0, high = 1, size = 1)
        self.correct = self.p_L * (1 - self.p_S) + (1 - self.p_L) * self.p_G

    def update(self, observation):
        if observation == 1:
            p_L_obs = self.p_L * (1 - self.p_S) / (self.p_L * (1 - self.p_S) + (1 - self.p_L) * self.p_G)
        else:
            p_L_obs = self.p_L * self.p_S / (self.p_L * self.p_S + (1 - self.p_L) * (1 - self.p_G))
        self.p_L = p_L_obs + (1 - p_L_obs) * self.p_T
        self.correct = self.p_L * (1 - self.p_S) + (1 - self.p_L) * self.p_G

    def answer(self):
        return np.random.binomial(n = 1, p = self.correct, size = 1)

    def reset(self):
        np.random.seed(100)
        self.p_L = np.random.uniform(low=0, high=1, size=1)
        self.p_T = np.random.uniform(low=0, high=1, size=1)
        self.p_S = np.random.uniform(low=0, high=1, size=1)
        self.p_G = np.random.uniform(low=0, high=1, size=1)
        self.correct = self.p_L * (1 - self.p_S) + (1 - self.p_L) * self.p_G



class BKTStudent:
    def __init__(self, num_skills):
        self.num_skills = num_skills
        self.knowledge_states = [BKTStudentSkill() for _ in range(num_skills)]

    def answer(self, skill_idx):
        return self.knowledge_states[skill_idx].answer()

    def updateKnowledge(self, observation, skill_idx):
        self.knowledge_states[skill_idx].update(observation)

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

s = BKTStudent(num_skills=3)
s.takePreTest()
