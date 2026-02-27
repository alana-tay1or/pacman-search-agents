# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()

        def minimax(state, agentIndex, depthReached):
            actions = state.getLegalActions(agentIndex)
            
            if state.isWin() or state.isLose() or depthReached == self.depth: #array slice doesn't work just write it out
                return self.evaluationFunction(state)

            agentSucc = (agentIndex + 1) % agentNum #to make sure that we're actually circling through agents lol
            if agentSucc == 0:
                newDepthReached = depthReached + 1
            else:
                newDepthReached = depthReached

            if agentIndex == 0:
                best = float('-inf')
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    v = minimax(nextNode, agentSucc, newDepthReached) #recursive :)
                    if v > best:
                        best = v
                return best        

            if agentIndex >= 1:
                best = float('inf')
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    v = minimax(nextNode, agentSucc, newDepthReached) #more recursion :)))
                    if v < best:
                        best = v
                return best

        root = float('-inf')
        lever = None
        for i in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, i)
            score = minimax(succ, 1, 0) #will the recursive function handle me just putting 1 lol
            if score > root:
                root = score
                lever = i

        return lever
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        #how can i get alpha & beta to be passed down through the entire func???

        def alphabeta(state, agentIndex, depthReached, a, b):
            actions = state.getLegalActions(agentIndex)
            
            if state.isWin() or state.isLose() or depthReached == self.depth:
                return self.evaluationFunction(state)

            agentSucc = (agentIndex + 1) % agentNum 
            if agentSucc == 0:
                newDepthReached = depthReached + 1
            else:
                newDepthReached = depthReached

            if agentIndex == 0:
                best = float('-inf')
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    v = alphabeta(nextNode, agentSucc, newDepthReached, a, b)
                    best = max(best, v)
                    a = max(a, best)
                    if a > b:
                        return best
                    
                return best        

            if agentIndex >= 1:
                best = float('inf')
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    v = alphabeta(nextNode, agentSucc, newDepthReached, a, b)
                    best = min(best, v)
                    b = min(b, best)
                    if a > b:
                        return best
                    
                return best

        root = float('-inf')
        lever = None
        a = float('-inf')
        b = float('inf')
        for i in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, i)
            score = alphabeta(succ, 1, 0, a, b)
            if score > root:
                root = score
                lever = i
            a = max(score, a)

        return lever

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()

        def expectimax(state, agentIndex, depthReached):
            actions = state.getLegalActions(agentIndex)
            
            if state.isWin() or state.isLose() or depthReached == self.depth:
                return self.evaluationFunction(state)

            agentSucc = (agentIndex + 1) % agentNum 
            if agentSucc == 0:
                newDepthReached = depthReached + 1
            else:
                newDepthReached = depthReached

            if agentIndex == 0:
                best = float('-inf')
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    v = expectimax(nextNode, agentSucc, newDepthReached)
                    if v > best:
                        best = v
                return best        

            if agentIndex >= 1:
                best = 0
                givenValue = 0
                for i in actions:
                    nextNode = state.generateSuccessor(agentIndex, i)
                    prob = 1 / len(actions)
                    v = expectimax(nextNode, agentSucc, newDepthReached)
                    givenValue += prob * v
                return givenValue

        root = float('-inf')
        lever = None
        for i in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, i)
            score = expectimax(succ, 1, 0)
            if score > root:
                root = score
                lever = i

        return lever

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    pacmanScore = currentGameState.getScore()

    foodList = newFood.asList()
    if foodList == []:
        foodScore = 0
    else:    
        foodDist = []   
        for i in foodList:
            distance = manhattanDistance(newPos, i)
            foodDist.append(distance)

        minFoodDist = min(foodDist)
        foodScore = 1 / (minFoodDist + 1)

    ghostScore = 0
    for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(newPos, ghostPos)
        if scaredTime > 0:
            ghostScore += 2 / (dist + 1)
        else:
            if dist < 2:
                ghostScore -= 10
            else:
                ghostScore -= 1 / (dist + 1)

    finalScore = pacmanScore + (10 * foodScore) + ghostScore

    return finalScore

# Abbreviation
better = betterEvaluationFunction
