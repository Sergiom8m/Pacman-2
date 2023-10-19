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


import random

import util
from game import Agent
from util import manhattanDistance


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        '''
        #Comida
        foodList=newFood.asList()
        minDistance=-1
        for food in foodList:
            distanciaF=util.manhattanDistance(food,newPos)
            if(minDistance>=distanciaF or minDistance==-1):
                minDistance=distanciaF


        #Fantasma
        for ghost in newGhostStates:
            distanciaG=util.manhattanDistance(ghost.getPosition(),newPos)
            # si hay mas?? distanciaSum=distanciaSum+distanciaG

        if(distanciaG>1):
            return successorGameState.getScore()+(1/minDistance)+(1/distanciaG)

        else:
            return successorGameState.getScore()+(1/minDistance)
        '''

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()

        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()

        newGhostStates = successorGameState.getGhostStates()
        oldGhostStates = currentGameState.getGhostStates()

        "*** YOUR CODE HERE ***"

        # CASOS EXTREMOS

        if successorGameState.isLose():
            return -100000
        elif successorGameState.isWin():
            return 100000

        # CALCULAR DISTANCIAS A LA COMIDA (ANTES VS DESPUES)

        newFoodDist = [util.manhattanDistance(newPos, food) for food in newFood]
        oldFoodDist = [util.manhattanDistance(oldPos, food) for food in oldFood]

        minNewFoodDist = min(newFoodDist)
        minOldFoodDist = min(oldFoodDist)

        # CALCULAR DISTANCIAS A LOS FANTASMAS (ANTES VS DESPUES)

        newGhostDistances = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        oldGhostDistances = [util.manhattanDistance(oldPos, ghostState.getPosition()) for ghostState in oldGhostStates]

        minNewGhostDist = min(newGhostDistances)
        minOldGhostDist = min(oldGhostDistances)

        # FACTOR 1 -> CERCANIA A LA COMIDA
        v1 = 0
        if newFoodDist and minNewFoodDist < minOldFoodDist:
            v1 = 40

        # FACTOR 2 -> NUMERO DE COMIDAS EN EL TABLERO
        v2 = 0
        if len(newFoodDist) < len(oldFoodDist):
            v2 = 170

        # FACTOR 3 -> CERCANIA A LOS FANTASMAS
        v3 = 0
        if minNewGhostDist > minOldGhostDist:
            v3 = 10
        else:
            v3 = -5

        value = v1 + v2 + v3

        return successorGameState.getScore() - currentGameState.getScore() + value


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
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
        """
        max_depth = 2

        def value(state, agentIndex):

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            elif agentIndex >= 1:
                min_value(state, agentIndex)

            else:
                max_value(state)

        def max_value(state):
            v = -10000
            legalActions = state.getLegalActions(0)
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                v = max(v, value(successor, 1))
            return v

        def min_value(state, agentIndex):
            v = 10000
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, value(successor, agentIndex + 1))
            return v

        legalActions = game_state.getLegalActions(0)
        successors = []
        for action in legalActions:
            successors.append(game_state.generateSuccessor(0, action))
        """

        def value(state, agentIndex, depth):

            # ESTADOS TERMINALES (GANAR/PERDER O PROFUNDIDAD ALCANZADA)
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # AGENTES FANTASMA
            elif agentIndex >= 1:
                return min_value(state, agentIndex, depth)

            # AGENTE PACMAN
            else:
                return max_value(state, depth)

        def max_value(state, depth):  # SIEMPRE SE EJECUTARA PARA PACMAN

            v = float('-inf')
            legalActions = state.getLegalActions(0)  # ACCIONES POSIBLES PARA PACMAN

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                v = max(v, value(successor, 1, depth))  # ELEGIR EL MAXIMO DE LOS SUCESORES

            return v

        def min_value(state, agentIndex, depth):  # SE EJECUTA EN LOS DIFERENTES FANTASMAS

            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)

                # SI ES EL ULTIMO FANTASMA, DESPUES PACMAN
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, value(successor, 0, depth - 1))

                # SI NO LE TOCARA A OTRO FANTASMA
                else:
                    v = min(v, value(successor, agentIndex + 1, depth))

            return v

        legal_actions = game_state.getLegalActions(0)  # OBTENER ACCIONES POSIBLES PARA PACMAN
        best_action = None
        best_value = float('-inf')

        # CALCULAR UN VALOR VALUE PARA CADA ACCION POSIBLE
        for action in legal_actions:
            sucessor = game_state.generateSuccessor(0, action)
            v = value(sucessor, 1, self.depth)
            if v > best_value:
                best_value = v
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def value(state, agentIndex, depth, alpha, beta):
            # terminal state
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # next agent is min (ghost)
            elif agentIndex >= 1:
                return min_value(state, agentIndex, depth, alpha, beta)
            # next agent is max (pacman)
            else:
                return max_value(state, depth, alpha, beta)

        def max_value(state, depth, alpha, beta):
            v = float('-inf')
            legalActions = state.getLegalActions(0)
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                v = max(v, value(successor, 1, depth, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v

        def min_value(state, agentIndex, depth, alpha, beta):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, value(successor, 0, depth - 1, alpha, beta))
                else:
                    v = min(v, value(successor, agentIndex + 1, depth, alpha, beta))
                if v < alpha: return v
                beta = min(beta, v)
            return v

        legal_actions = game_state.getLegalActions(0)
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in legal_actions:
            sucessor = game_state.generateSuccessor(0, action)
            v = value(sucessor, 1, self.depth, alpha, beta)
            if v > best_value:
                best_value = v
                best_action = action
            alpha = max(alpha, best_value)  # v1 sin esto --> 0/5
        return best_action


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
