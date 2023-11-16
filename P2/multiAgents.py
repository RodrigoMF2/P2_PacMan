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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        """
        Projete uma função de avaliação melhor aqui.

        A função de avaliação inclui o sucessor atual e proposto
        GameStates (pacman.py) e retorna um número, onde números mais altos são melhores.

        O código abaixo extrai algumas informações úteis do estado, como a
        comida restante (newFood) e posição do Pacman após a movimentação (newPos).
        newScaredTimes contém o número de movimentos que cada fantasma permanecerá
        assustado porque Pacman comeu uma bolinha de energia.

        Imprima essas variáveis ​​para ver o que você está obtendo e combine-as
        para criar uma função de avaliação magistral.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood() # newFood é uma lista list()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # criamos uma lista com a posiçao de todos os ghost
        position_ghosts = [ghost for ghost in successorGameState.getGhostPositions()]
        # lista com as distancias do pacmam para cada alimento
        distances = [manhattanDistance(newPos,food) for food in newFood.asList()]

        # se nao houver alimento retorna infinto
        if len( distances) == 0:
            return float('inf')
        extrascore = 0.0

        """Encontra a menor distância entre o Pac-Man e qualquer comida na lista,e depois 
        adiciona um extrascore, quanto menor a distancia, maior a pontuação """

        min_distance = min(distances)
        extrascore = 1/min_distance * 10
        """percorre a lista de posiçoes dos ghost e se o fantasma estiver assutado"""
        for index,ghost in enumerate(position_ghosts):
            if newScaredTimes[index] > 0: # Se o fantasma estiver assustado, ignora-o, pois não representa uma ameaça imediata.
                continue
            else:
                if manhattanDistance(ghost,newPos) < 2:
                    extrascore = float('-inf')
        return successorGameState.getScore() + extrascore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        legalActions = gameState.getLegalActions(0)  # 0 represents Pacman

        # Inicializa as variáveis para armazenar a melhor ação e o melhor valor
        bestAction = None
        bestValue = float("-inf")

        # Itera sobre as ações legais e escolhe a que maximiza o valor Minimax
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.min_value(successor, self.depth, 1)  # Começa na camada Max (Pacman)

            # Atualiza a melhor ação se encontrar um valor melhor
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

    def min_value(self, state, depth, agentIndex):
        """
        Função auxiliar para o algoritmo Minimax, representa a camada Min.
        """
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        numAgents = state.getNumAgents()

        # Inicializa o valor com infinito positivo, pois estamos minimizando
        value = float("inf")

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)

            # Se todos os fantasmas se moveram, passa para a próxima camada Max (Pacman)
            if agentIndex == numAgents - 1:
                value = min(value, self.max_value(successor, depth - 1))
            else:
                value = min(value, self.min_value(successor, depth, agentIndex + 1))

        return value

    def max_value(self, state, depth):
        """
        Função auxiliar para o algoritmo Minimax, representa a camada Max.
        """
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(0)  # Ações do Pacman

        # Inicializa o valor com infinito negativo, pois estamos maximizando
        value = float("-inf")

        for action in legalActions:
            successor = state.generateSuccessor(0, action)
            value = max(value, self.min_value(successor, depth, 1))  # Próxima camada é a Min (fantasmas)

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        Infinity = float('inf')

        def minValue(state, agentIndex, depth, alpha, beta):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            #Inicializer o valor com valor para infinito positivo
            v = Infinity
            #Itera sobre as ações legais
            for action in legalActions:
                newState = state.generateSuccessor(agentIndex, action)

                #Se for o ultimo agente, chama a função maxValue, senão minValue
                if agentIndex == state.getNumAgents() - 1:
                    newV = maxValue(newState, depth, alpha, beta)
                else:
                    newV = minValue(newState, agentIndex + 1, depth, alpha, beta)

                v = min(v, newV)
                #Podagem alpha-beta: se o valor é menor que alpha,retorna o valor
                if v < alpha:
                    return v
                #atualiza beta para o minimo entre beta e o balor atual
                beta = min(beta, v)
            return v

        def maxValue(state, depth, alpha, beta):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            #Inicializa com valor para infinito negativo
            v = -Infinity
            # se a pronfudidade é 0,  retorna a primeira ação legal
            if depth == 0:
                bestAction = legalActions[0]
            #Itera sobre as ações legais
            for action in legalActions:
                newState = state.generateSuccessor(0, action)
                #Chama minValue para os fantasmas subsequentes
                newV = minValue(newState, 0 + 1, depth + 1, alpha, beta)
                #Atualiza o valor maximo e a melhor açao
                if newV > v:
                    v = newV
                    if depth == 0:
                        bestAction = action

                #Podagem alpha-beta: se o valor é maior que beta,retorna o valor
                if v > beta:
                    return v
                #atualiza alpha para o maximo entre alpha e o valor atual
                alpha = max(alpha, v)

            #Se a profundidade for 0, retorna a melhor ação
            if depth == 0:
                return bestAction
            return v

        bestAction = maxValue(gameState, 0, -Infinity, Infinity)
        return bestAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expValue( state, agentIndex, depth):
            # information about the agent count and the legal actions for the index
            agentCount = gameState.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            # if no legal actions then return the evaluation function
            if not legalActions:
                return self.evaluationFunction(state)

            # expected value and the probabilyt
            expectedValue = 0
            probabilty = 1.0 / len(legalActions)  # probability of each action
            # pacman is the last to move after all ghost movement
            for action in legalActions:
                if agentIndex == agentCount - 1:
                    currentExpValue = maxValue(state.generateSuccessor(agentIndex, action), \
                                               agentIndex, depth)
                else:
                    currentExpValue = expValue(state.generateSuccessor(agentIndex, action), \
                                               agentIndex + 1, depth)
                expectedValue += currentExpValue * probabilty

            return expectedValue

        # maximum value function used for only pacman and hence setting index to 0
        def maxValue(state, agentIndex, depth):
            # information about the agent index and the legal actions for the index
            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex)

            # if no legal actions or depth reached(prevent maximum depth
            # exceeded in recursion)then return the evaluation function
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            maximumValue = max(expValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in legalActions)

            return maximumValue

        # maximizing the best possible moves for the rootnode i.e.
        # the pacman thus agent index 0
        actions = gameState.getLegalActions(0)
        # find all actions and the corresponding value and then return action
        # corresponding to the maximum value
        allActions = {}
        for action in actions:
            allActions[action] = expValue(gameState.generateSuccessor(0, action), 1, 1)

        # returning action with best expectimax value
        return max(allActions, key=allActions.get)
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
 DESCRIÇÃO: Para construir uma função de avaliação melhor, eu
    reprogramei a função de avaliação inicial e a melhorei. Para o estado atual,
    os melhores e piores casos são identificados, ou seja, vencer e
    estar no mesmo estado que um fantasma não assustado, a pontuação de 99999
    e -99999, respectivamente.

    Para devorar comida, quanto mais próxima a comida melhor e menos pellets de comida
    restantes é um ponto positivo.

    Para pegar pellets, quanto mais perto os pellets melhor a pontuação.

    Para caçar fantasmas, utilizei a soma dos tempos assustados para verificar se
    o tempo assustado está ocorrendo e, se sim, quanto mais perto o fantasma melhor,
    caso contrário, a pontuação é ruim.
    """
    "*** YOUR CODE HERE ***"
    # Informações úteis que podem ser extraídas de um GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsule = currentGameState.getCapsules()

    # Pontuação mais alta para um estado vencedor
    if currentGameState.isWin():
        return 9999

    # Pior caso se a posição do Pac-Man e do fantasma forem iguais
    # mas o fantasma não estiver assustado
    for state in currentGhostStates:
        if state.getPosition() == currentPos and state.scaredTimer == 1:
            return -9999

    score = 0

    # Caça a comida - devoração de comida
    # Pontuação melhor para estados com comida perto e fantasmas longe
    # Verifica a distância da comida até o Pac-Man
    foodDistance = [util.manhattanDistance(currentPos, food) for food in currentFood]
    nearestFood = min(foodDistance)
    # Comida mais próxima deve ter mais peso - usa o inverso
    score += float(1 / nearestFood)
    # Subtrai o número de comidas restantes e peso proporcional, pois queremos escolher
    # estados com menos comida restante
    score -= len(currentFood)

    # Caça a cápsulas - coleta de pellets
    # Pontuação para cápsulas
    if currentCapsule:
        capsuleDistance = [util.manhattanDistance(currentPos, capsule) for capsule in currentCapsule]
        nearestCapsule = min(capsuleDistance)
        # Cápsula mais próxima é melhor
        score += float(1 / nearestCapsule)

    # Caça fantasmas quando o fantasma está assustado, caso contrário, evita - caça a fantasmas
    currentGhostDistances = [util.manhattanDistance(currentPos, ghost.getPosition()) for ghost in
                             currentGameState.getGhostStates()]
    nearestCurrentGhost = min(currentGhostDistances)
    scaredTime = sum(currentScaredTimes)
    # Fantasmas mais distantes são melhores
    if nearestCurrentGhost >= 1:
        if scaredTime < 0:
            score -= 1 / nearestCurrentGhost
        else:
            score += 1 / nearestCurrentGhost

    return currentGameState.getScore() + score


# Abbreviation
better = betterEvaluationFunction
