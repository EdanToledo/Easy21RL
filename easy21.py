import numpy as np

class easy21:
    def __init__(self):
        self.minCardValue = 1
        self.maxCardValue = 10
        self.lowerBound = 0
        self.upperBound = 21
        self.dealerCutoff = 17
        self.redProb = 1/3
        self.hit = 0
        self.stick =1
       
    def draw(self):
        value = np.random.randint(self.minCardValue, self.maxCardValue+1)
        if np.random.uniform(0, 1) <= self.redProb:
            return -value
        else:
            return value

    def startGame(self):
        return (np.random.randint(self.minCardValue, self.maxCardValue+1), np.random.randint(self.minCardValue, self.maxCardValue+1))

    def bust(self, value):
        if value > self.upperBound or value < self.lowerBound:
            return True
        else:
            return False

    def step(self, player, dealer, action):
        if action == self.hit:

            player += self.draw()

            if self.bust(player):
                reward = -1
                terminated = True
            else:
                reward = 0
                terminated = False

        elif action == self.stick:
            terminated = True

            while self.lowerBound < dealer < self.dealerCutoff:
                dealer += self.draw()

            if self.bust(dealer) or player > dealer:
                reward = 1
            elif player == dealer:
                reward = 0
            elif player < dealer:
                reward = -1

        return player, dealer, reward, terminated
