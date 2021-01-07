import numpy as np
from easy21 import easy21
import csv
import pickle


class player:
    def __init__(self, tdLambda,epsilon,stepSize):
        self.actions = [0, 1]
        self.weights = np.zeros((36, 1))
        self.eligibilityTrace = np.zeros((36, 1))
        self.game = easy21()
        self.tdLambda = tdLambda
        self.epsilon = epsilon
        self.stepSize = stepSize

    def chooseAction(self, d, p):
        #exploratory Action
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
        else:
        #Greedy action
            action = np.argmax([self.qValue(d, p, a) for a in self.actions])
        return action

    #Outputs value function for states
    def OptimalValueFunction(self):
        v = {}
        for i in range(1, 11):
            for j in range(1, 22):
                v[i, j] = max(self.qValue(i, j, 0), self.qValue(i, j, 1))
        return v

    #creates Feature vector
    def featureVector(self, d, p, a):
        f = np.zeros((3, 6, 2))
        dp = []
        for index, (l, u) in enumerate(zip(range(1, 8, 3), range(4, 11, 3))):
            if (l <= d <= u):
                dp.append(index)
        pp = []
        for index, (l, u) in enumerate(zip(range(1, 17, 3), range(6, 22, 3))):
            if (l <= p <= u):
                pp.append(index)

        for i in dp:
            for j in pp:
                f[i, j, a] = 1

        return f.reshape(1, -1)

    #Linear function approximator for state-action value
    def qValue(self, d, p, a):
        q = self.featureVector(d, p, a).dot(self.weights)
        return q[0][0]

    #Trains agent
    def playLinearFunctionApprox(self, NumEpisodes, trueQ):
        wins = 0
        meanReturn = 0
        for z in range(NumEpisodes):

            terminated = False
            self.eligibilityTrace = np.zeros((36, 1))

            d, p = self.game.startGame()
            a = self.chooseAction(d, p)
            while not terminated:

                p_next, d_next, r, terminated = self.game.step(p, d, a)

                if not terminated:
                    a_next = self.chooseAction(d_next, p_next)
                    delta = r+self.qValue(d_next, p_next,
                                          a_next)-self.qValue(d, p, a)
                else:
                    delta = r-self.qValue(d, p, a)

                self.eligibilityTrace = self.tdLambda*self.eligibilityTrace + \
                    self.featureVector(d, p, a).reshape(36, -1)
                change = self.stepSize*delta*self.eligibilityTrace
                self.weights += change

                if not terminated:
                    d, p, a = d_next, p_next, a_next

            meanReturn = meanReturn + 1/(z+1) * (r - meanReturn)
            if r == 1:
                wins += 1

            if (((z) % 1000) == 0):
                print("Episode %i, Mean-Return %.3f, MSE %.3f, Wins %.3f" %
                      (z, meanReturn, self.calculateMSE(trueQ), wins/(z+1)))
    
    #Calculates MSE of current Q function vs some true Q function
    def calculateMSE(self, trueQ):
        MSE = 0
        for i in range(1, 11):
            for j in range(1, 22):
                for k in range(0, 2):
                    MSE += pow(self.qValue(i, j, k) - trueQ[i, j, k], 2)

        return MSE/(10*21*2)

    #Creates CSV of Value Function
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
        p = player(i/10,0.05,0.01)
        p.playLinearFunctionApprox(10000, TrueQ)
        print("Lambda", i/10, "finished")
        mse.append(p.calculateMSE(TrueQ))
    print("final MSE:")
    print(mse)
