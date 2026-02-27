# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            update_values_dict = self.values.copy()

            # get Q_values for each possible s_prime
            for state in self.mdp.getStates():
                Q_values = [float('-inf')]
                terminal_state = self.mdp.isTerminal(state) # boolean
                
                #terminal states have 0 value
                if terminal_state:
                    update_values_dict[state] = 0
                
                else:
                    legal_actions = self.mdp.getPossibleActions(state)
                    for action in legal_actions:
                        Q_values.append(self.getQValue(state, action))

                    #update value function at state s to largest Q_value
                    update_values_dict[state] = max(Q_values)
            self.values = update_values_dict

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        for i, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qVal += prob*(self.mdp.getReward(state, action, i) + self.discount*self.values[i])
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action = self.mdp.getPossibleActions(state)
        if len(action) == 0:
            return None
        action = []

        bestQ = float("-inf")
        for i in self.mdp.getPossibleActions(state):
            qVal = self.computeQValueFromValues(state, i)
            if(qVal > bestQ):
                bestQ = qVal
                action = i
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num = len(states)
        index = 0

        for i in range(self.iterations):
            state = states[index]

            if not self.mdp.isTerminal(state):
                best = self.computeActionFromValues(state)
                if best is not None:
                    q_value = self.computeQValueFromValues(state, best)
                    self.values[state] = q_value
            index = (index+1) % num

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {s: set() for s in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[next].add(state)
        # priority queue
        pq = util.PriorityQueue()
        # Initialize priorities and push all non-terminal states into the priority queue.
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                possible = self.mdp.getPossibleActions(state)
                maxqVal = float('-inf')
                for action in possible:
                    currqVal = self.computeQValueFromValues(state, action)
                    if currqVal > maxqVal:
                        maxqVal = currqVal
                diff = abs(self.values[state] - maxqVal)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()

            if not self.mdp.isTerminal(s):
                possible = self.mdp.getPossibleActions(s)
                maxqVal = float('-inf')
                for action in possible:
                    currqVal = self.computeQValueFromValues(s, action)
                    if currqVal > maxqVal:
                        maxqVal = currqVal
                self.values[s] = maxqVal
            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    possible = self.mdp.getPossibleActions(p)
                    maxqVal = float('-inf')
                    for action in possible:
                        currqVal = self.computeQValueFromValues(p, action)
                        if currqVal > maxqVal:
                            maxqVal = currqVal
                    diff = abs(self.values[p] - maxqVal)
                    if diff > self.theta:
                        pq.update(p, -diff)