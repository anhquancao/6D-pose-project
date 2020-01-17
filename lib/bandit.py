import numpy as np
import scipy.stats as ss
from scipy.stats import beta


class TS:
    def __init__(self, nbArms, maxReward=1.):
        self.A = nbArms
        self.clear()

    def clear(self):
        self.NbPulls = np.zeros(self.A)
        self.params = [(2, 2) for i in range(self.A)]

    def chooseArmToPlay(self):
        for a in range(self.A):
            if self.NbPulls[a] == 0:
                return a
        samples = []
        for arm in range(self.A):
            a, b = self.params[arm]
            sample = beta.rvs(a, b, size=1)
            samples.append(sample)
        return np.argmax(samples)

    def receiveReward(self, arm, reward):
        self.NbPulls[arm] = self.NbPulls[arm] +1
        a, b = self.params[arm]
        a = a + reward
        b = b + 1 - reward
        self.params[arm] = (a, b)

    def name(self):
        return "TS"