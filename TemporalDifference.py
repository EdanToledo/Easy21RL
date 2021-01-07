import numpy as np
from easy21 import easy21
import csv
import pickle


class player:
    def __init__(self, N_0, tdLambda):
        self.N_0 = N_0
        self.actions = [0, 1]
        self.Q = np.zeros((11, 22, len(self.actions)))
        self.eligibilityTrace = np.zeros((11, 22, len(self.actions)))
        self.NSA = np.zeros((11, 22, len(self.actions)))
        self.game = easy21()
        self.tdLambda = tdLambda

    def NS(self, d, p):
        return np.sum(self.NSA[d, p])

    def epsilon(self, d, p):
        return self.N_0/(self.N_0+self.NS(d, p))

    def stepSize(self, d, p, a):
        return 1/self.NSA[d, p, a]

    def chooseAction(self, d, p):
        if np.random.uniform(0, 1) < self.epsilon(d, p):
            action = np.random.choice(self.actions)
        else:
            action = np.argmax([self.Q[d, p, a] for a in self.actions])
        return action

    def OptimalValueFunction(self):
        v = {}
        for i in range(1, 11):
            for j in range(1, 22):
                v[i, j] = max(self.Q[i, j, 0], self.Q[i, j, 1])
        return v

    def playTemporalDifference(self, NumEpisodes):

        for z in range(NumEpisodes):
            visited = {}
            terminated = False
            self.eligibilityTrace = np.zeros((11, 22, len(self.actions)))

            d, p = self.game.startGame()
            a = self.chooseAction(d, p)
            while not terminated:

                p_next, d_next, r, terminated = self.game.step(p, d, a)
                if not terminated:
                    a_next = self.chooseAction(d_next, p_next)
                    delta = r+self.Q[d_next, p_next, a_next]-self.Q[d, p, a]
                else:
                    delta = r-self.Q[d, p, a]
                self.NSA[d, p, a] += 1
                self.eligibilityTrace[d, p, a] += 1
                visited[d, p, a] = True

                for (i, j, k) in visited:
                    self.Q[i, j, k] += self.stepSize(i, j, k) * \
                        delta*self.eligibilityTrace[i, j, k]
                    self.eligibilityTrace[i, j, k] *= self.tdLambda

                if not terminated:
                    d, p, a = d_next, p_next, a_next

    def calculateMSE(self, trueQ):
        MSE = 0
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    MSE += pow(self.Q[i, j, k] - trueQ[i, j, k], 2)

        return MSE/(10*21*2)

    def outputValueCSV(self, v):
        with open('valueFunction.csv', mode='w') as csv_file:
            value_writer = csv.writer(
                csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            value_writer.writerow(['dealer', 'player', 'value'])
            for i in range(1, 11):
                for j in range(1, 22):
                    value_writer.writerow([i, j, v[i, j]])


if __name__ == "__main__":

    mse = []

    with open('Q.mc', 'rb') as Qfile:
        TrueQ = pickle.load(Qfile)

    for i in range(0, 11):
        p = player(100, i/10)
        p.playTemporalDifference(10000)
        print("Lambda", i/10, "finished")
        mse.append(p.calculateMSE(TrueQ))
    
    print(mse)
