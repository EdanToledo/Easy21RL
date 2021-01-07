import numpy as np
from easy21 import easy21
import csv
import pickle


class player:
    def __init__(self, N_0):
        self.N_0 = N_0
        self.actions = [0, 1]
        self.Q = np.zeros((11, 22, len(self.actions)))
        self.NSA = np.zeros((11, 22, len(self.actions)))
        self.game = easy21()

    def NS(self, state):
        (d, p) = state
        return np.sum(self.NSA[d, p])

    def epsilon(self, state):
        return self.N_0/(self.N_0+self.NS(state))

    def stepSize(self, SA):
        return 1/self.NSA[SA]

    def chooseAction(self, state):
        if np.random.uniform(0, 1) < self.epsilon(state):
            action = np.random.choice(self.actions)
        else:
            (d, p) = state
            action = np.argmax([self.Q[d, p, a] for a in self.actions])
        return action

    def OptimalValueFunction(self):
        v = {}
        for i in range(1, 11):
            for j in range(1, 22):
                v[i, j] = max(self.Q[i, j, 0], self.Q[i, j, 1])
        return v

    def updateQ(self, record):
        G = sum([exp[-1] for exp in record])
        for d, p, a, r in record:
            self.Q[d, p, a] += self.stepSize((d, p, a))*(G-self.Q[d, p, a])

    def playMonteCarlo(self, NumEpisodes):
        currentEp = 1
        terminated = False
        episodeRecord = []
        wins = 0
        meanReturn = 0
        while currentEp <= NumEpisodes:

            dealer, player = self.game.startGame()

            while not terminated:

                action = self.chooseAction((dealer, player))

                self.NSA[dealer, player, action] += 1

                player_new, dealer_new, reward, terminated = self.game.step(player, dealer, action)

                episodeRecord.append([dealer, player, action, reward])
                dealer, player = dealer_new, player_new

            meanReturn = meanReturn + 1/(currentEp) * (reward - meanReturn)
            if reward == 1:
                wins += 1

            if currentEp % 10000 == 0:
                print("Episode %i, Mean-Return %.3f, Wins %.3f" %
                      (currentEp, meanReturn, wins/(currentEp)))
            self.updateQ(episodeRecord)
            currentEp += 1
            terminated = False
            episodeRecord = []

    def outputValueCSV(self, v):
        with open('valueFunction.csv', mode='w') as csv_file:
            value_writer = csv.writer(
                csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            value_writer.writerow(['dealer', 'player', 'value'])
            for i in range(1, 11):
                for j in range(1, 22):
                    value_writer.writerow([i, j, v[i, j]])


if __name__ == "__main__":
    p = player(100)
    p.playMonteCarlo(2000000)
    v = p.OptimalValueFunction()
    with open('Q.mc', 'wb') as Qfile:
        pickle.dump(p.Q, Qfile)
    p.outputValueCSV(v)