import numpy as np
from collections import deque

class Agent(object): #Keep the class name!
    def __init__(self, state_space, action_space, gamma = 0.9, alpha=0.1, N=10, n=5, init=25):
        self.action_space = action_space # integer
        self.state_space = state_space # integer

        self.gamma = gamma
        self.alpha = alpha
        self.Q1 = init*np.ones((state_space, action_space))
        self.Q2 = init*np.ones_like(self.Q1)

        # avgRew[s, a] = average reward we've gotten when taking action a from state s
        self.avgRew = np.zeros_like(self.Q1)

         # stateActFreq[s, a] = how many times we've been in state s and taken action a
        self.stateActFreq = np.zeros_like(self.Q1)
        
        # stateTransFreq[s, a, s'] = frequency of transition (s, a) -> s'
        self.stateTransFreq = np.zeros((state_space, action_space, state_space))

        self.N = N # no. of steps in planning phase
        self.n = n # no. of steps in return computation (n-step return)
        self.rewards = deque() # (up to) n last rewards seen
        self.stateActionHist = deque() # (up to) n last (s, a) seen

        self.eps = 1
        self.t = 0

        self.maxDelta = 10
        self.lastDelta = self.maxDelta * np.ones_like(self.Q1) # TD error for last update of each Q(s, a)

    def observe(self, observation, reward, done):
        self.rewards.append(reward)
        #If we find a terminating state, we can't get more rewards from that state.
        if done:
            self.Q1[observation,:] = np.zeros(self.action_space)
            self.Q2[observation,:] = np.zeros(self.action_space)
        # determine whether Q should be updated yet, depending on whether we have observed n rewards yet (for n-step return)
        updateQ = len(self.rewards) >= self.n

        # update Q(s, a) with data (s', r)
        if updateQ:
            cumRew = self.computeMultiStepReturn()
            self.rewards.popleft()
            originalStateAction = self.stateActionHist.popleft()
            startState = originalStateAction[0]
            startAction = originalStateAction[1]
            # update either Q1 or Q2 with 50 % chance
            if np.random.rand() <= 0.5:
                delta = cumRew + self.gamma **len(self.rewards) * self.Q2[observation, np.argmax(self.Q1[observation, :])] - self.Q1[startState, startAction]
                self.Q1[startState, startAction] += self.alpha * delta
            else:
                delta = cumRew + self.gamma **len(self.rewards)* self.Q1[observation, np.argmax(self.Q2[observation, :])] - self.Q2[startState, startAction]
                self.Q2[startState, startAction] += self.alpha * delta
            self.lastDelta[startState, startAction] = np.abs(delta)

        # update Model(s, a) with (s', r)
        lastState = self.stateActionHist[-1][0]
        lastAction = self.stateActionHist[-1][1]
        self.stateTransFreq[lastState, lastAction, observation] += 1
        self.avgRew[lastState, lastAction] += (reward - self.avgRew[lastState, lastAction]) / self.stateActFreq[lastState, lastAction]

        # planning phase: update Q1/Q2 N times (1-step error instead of n-step)
        for _ in range(self.N):
            sum = self.lastDelta.sum()
            if sum == 0:
                s = np.random.choice(self.state_space)
                a = np.random.choice(self.action_space)
            else:
                idx = np.random.choice(self.state_space*self.action_space, p=(self.lastDelta/sum).flatten())
                s = int(np.floor(idx / self.action_space))
                a = idx % self.action_space

            # query model and update Q1 or Q2 with observation data (s', r)
            nextState, reward = self.queryModel(s, a)
            if np.random.rand() <= 0.5:
                delta = reward + self.gamma * self.Q2[nextState, np.argmax(self.Q1[nextState, :])] - self.Q1[s, a]
                self.Q1[s, a] += self.alpha * delta
            else:
                delta = reward + self.gamma * self.Q1[nextState, np.argmax(self.Q2[nextState, :])] - self.Q2[s, a]
                self.Q2[s, a] += self.alpha * delta
            self.lastDelta[s, a] = np.abs(delta)
        
        if done:
            self.reset()

        return 0

    def queryModel(self, state, action):
        # given (state, action), return nextState according to its relative frequency after (state, action) in the history
        # if (state, action) has never been seen before, return a random next state
        if sum(self.stateTransFreq[state, action, :]) == 0:
            nextState = np.random.randint(self.state_space)
        else:
            nextState = np.random.choice(self.state_space, p=self.stateTransFreq[state, action, :]/sum(self.stateTransFreq[state, action, :]))
        reward = self.avgRew[state, action]
        return nextState, reward

    def computeMultiStepReturn(self):
        ret = 0
        for i, reward in enumerate(self.rewards):
            ret += reward * self.gamma ** i
        return ret

    def act(self, observation):
        # epsilon greedy policy
        if np.random.rand() <= self.eps:
            action = np.random.randint(self.action_space)
        else:
            if np.random.rand() <= 0.5: q = self.Q1[observation, :]
            else:                       q = self.Q2[observation, :]
            action = np.random.choice(np.flatnonzero(q == q.max()))

        # update epsilon
        self.t += 1
        self.eps = 1 / self.t ** 0.3
        
        self.stateActFreq[observation, action] += 1
        self.stateActionHist.append([observation, action])

        return action

    def reset(self):
        #Update states that have been visited before terminating state.
        while self.rewards:
            cumRew = self.computeMultiStepReturn()
            self.rewards.popleft()
            originalStateAction = self.stateActionHist.popleft()
            startState = originalStateAction[0]
            startAction = originalStateAction[1]
            # update either Q1 or Q2 with 50 % chance
            if np.random.rand() <= 0.5:
                delta = cumRew - self.Q1[startState, startAction]
                self.Q1[startState, startAction] += self.alpha * delta
            else:
                delta = cumRew - self.Q2[startState, startAction]
                self.Q2[startState, startAction] += self.alpha * delta
            self.lastDelta[startState, startAction] = np.abs(delta)
        #Perhaps not needed but just in case.
        self.rewards.clear()
        self.stateActionHist.clear()