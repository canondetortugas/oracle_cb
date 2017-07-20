import numpy as np
import sys

"""
Module of bandit algorithms

Bandit algorithms should implement the following methods:
1. __init__(B): constructor that takes a Bandit Simulator object.
2. init(T): prepare to run for T rounds, wipe state, etc.
3. updated(x,a,r): update any state using the current interaction
4. get_action(x): propose an action for this context.
"""
class Bandit(object):
    """
    Bandit Algorithm interface. 

    This is a valid bandit algorithm that just plays randomly every
    round.
    """
    def __init__(self, B):
        self.B = B

    def init(self,T):
        self.reward = 0.0
        self.opt_reward = 0.0
        self.dist = [1.0/self.B.N for i in range(self.B.N)]

    def play(self,T):
        self.init(T)
        scores = []
        for t in range(T):
            if np.log2(t+1) == int(np.log2(t+1)):
                print("t = %d, r = %0.3f, ave_regret = %0.3f" % (t, self.reward, (self.opt_reward - self.reward)/(t+1)))
            x = self.B.get_new_context()
            p = self.get_action(x) 
            self.reward += self.B.get_reward(p)
            self.opt_reward += self.B.get_reward(self.B.Pi[self.B.Pistar,x])
            self.update(x, p, self.B.get_reward(p))
            scores.append(self.opt_reward - self.reward)
        return scores

    def update(self, x, a, r):
        pass

    def get_action(self, x):
        dist = [1.0/self.B.K for i in range(self.B.K)]
        p = np.random.multinomial(1, dist) ## Distribution over ACTIONS
        p = int(np.nonzero(p)[0])
        return p