import math

from domineering_game import *

class Node:
    def __init__(self,move,game,parent=None):
        self.move = move
        if parent is None:
            #current player
            self.player = game[1]
        else:
            self.player = not parent.player
        self.game = game
        self.parent = parent
        self.children = []
        self.score = 0 #probability of vertical player winning
        self.likelihood = 1
        self.likelihood_r = -1 #likelihood ratio

    def add_child(self,childNode):
        self.children.append(childNode)

    def update(self):
        if len(self.children) > 0:
            current_score = self.score
            if not self.game[1]:
                #first player wants to maximize score
                self.score = max([child.score for child in self.children])
            else:
                #second player wants to minimize score
                self.score = min([child.score for child in self.children])
            if self.score != current_score and self.parent:
                self.parent.update()

    def moves_sequence(self):
        if self.parent:
            moves = self.parent.moves_sequence()
            moves.append(self.move)
            return moves
        return []

    def eval_score(self):
        if self.likelihood_r != -1:
            #has already been calculated
            return self.likelihood_r
        if self.parent:
            self.likelihood_r = self.likelihood*self.parent.likelihood_r
            return self.likelihood_r
        else:
            self.likelihood_r = 1
            return 1

def MCTS(model,root,game,iterations=100,noise=0.1,likelihood_modifier=1):
    '''
    Performs a Monte Carlo Tree Search on a game using a neural network.
    iterations: number of positions to explore
    '''

    #first, figure out all non-terminal leaves of the tree stemming from the root
    #allows us to pass in a partially-evaluated tree
    unchecked = [root]
    to_expand = []
    posns = 0
    while len(unchecked) > 0:
        nextNode = unchecked.pop()
        if len(nextNode.children) > 0:
            #node we're currently checking isn't a leaf
            unchecked += nextNode.children
        else:
            if not nextNode.game[3]:
                #otherwise, the next node is terminal
                to_expand.append(nextNode)
        nextNode.likelihood_r *= likelihood_modifier #makes partial tree passing work
        posns += 1

    while posns < iterations and len(to_expand) > 0:
        # Batch evaluate nodes instead of individual evaluations
        eval_scores = [node.eval_score() for node in to_expand]
        noise_array = noise * np.random.rand(len(to_expand))
        eval_scores_with_noise = eval_scores + noise_array
        max_index = np.argmax(eval_scores_with_noise)
        node = to_expand.pop(max_index) #node which, up to randomness, is best to explore next

        # Optimize this part to batch model predictions if possible
        prediction = model.predict([[representation(node.game)]])
        #Input to model: [nxchannelsxBOARD_SIZExBOARD_SIZE] (in this case, n=1 bc predicting for 1 game, and channels=1)
        #Output from model: ([nx1],[nxN_MOVES])
        #Should this be [[representation(node.game)]]? CHECK AND TEST
        score, likelihoods = prediction[0],prediction[1]
        node.score = score[0]
        possible_continuations = []
        for i in range(112):
            if legal_moves(node.game)[i] == 1:
                #possible continuations contains all legal moves from the position at the node
                possible_continuations.append(i)

        possible_continuation_boards = []
        possible_continuation_reps = []
        moves_made = []
        for move in possible_continuations:
            next_game = copy_game(node.game)
            make_move(next_game,move)
            possible_continuation_boards.append(next_game)
            possible_continuation_reps.append([representation(next_game)])
            moves_made.append(move)

        scores = np.zeros(N_MOVES)
        evals = model.predict(possible_continuation_reps)[0]
        #can we store the rest of this evaluation on the node? Would save time, since we're double-evaluating if we expand this node later
        #Shape of evals: [nx1]
        for i in range(len(moves_made)):
            scores[moves_made[i]] = evals[i][0]

        for move_index in range(len(possible_continuations)):
            move = possible_continuations[move_index]
            newNode = Node(move, possible_continuation_boards[move_index], node)
            newNode.likelihood = likelihoods[0][move]
            newNode.score = scores[move]
            node.add_child(newNode)
            if not newNode.game[3]:
                to_expand.append(newNode)
            else:
                #game is over
                #If player False=0 wins, score is 1 and vice-versa
                newNode.score = float(not newNode.game[2])

        node.update()

        posns += len(node.children)

    return root

def self_play(model,game,iterations=500,noise=0.1,verbose=False):
    '''
    Performs self-play training on a neural network. Assume game is an initial position
    iterations: number of positions to explore
    '''
    move_history = []
    rep_history = []
    root = Node(None,game)
    root_likelihood = 1
    while not game[3]:
        root = MCTS(model,root,game,iterations,noise,likelihood_modifier=root_likelihood)
        move = None
        bestIndex = None
        if not game[1]:
            #Player 0
            index, child_node = max(enumerate(root.children), key=lambda x: x[1].score)
            move = child_node.move
            bestIndex = index
        else:
            #Player 1
            index, child_node = min(enumerate(root.children), key=lambda x: x[1].score)
            move = child_node.move
            bestIndex = index
        move_history.append(move)
        rep_history.append(np.copy(representation(game)))
        make_move(game,move)
        root = root.children[bestIndex]
        root_likelihood = 1/root.eval_score()
        if verbose:
            display(game)
    return (move_history, rep_history, game[2])

def training_examples(move_history,rep_history,winner):
    '''
    Generates training examples from a game.
    Returns: tuple (reps, scores, moves)
    Remember, winner 0 is score 1
    '''
    reps = []
    moves = []
    winners = []
    for i in range(len(move_history)):
        reps.append([rep_history[i]])
        move = move_history[i]
        move_vector = np.zeros(112)
        move_vector[move] = 1
        moves.append(move_vector)
        winners.append([not winner])
    return reps, moves, winners

def Arena(model1,model2,n,game,iterations,noise):
    '''
    Plays 2n games between two models, alternating sides
    Return 2 if model2 wins
    Uses double the iterations and one tenth the noise for MCTS
    '''
    #want the new model to be better than the old with p<0.05
    #so critical value 1.64
    #so threshold for games won is 2n*(0.5 + 1.62*sqrt(0.25/2n))
    #so threshold for score is 6.48n*sqrt(0.25/2n)
    score_threshold = math.floor(6.48*n*math.sqrt(0.125/n))

    score = 0
    game_data = []
    for i in range(n):
        #old model plays white
        reset(game)
        game_reps = []
        game_moves = []
        root1 = Node(None,game)
        root2 = Node(None,game)
        likelihood_ratio1 = 1
        likelihood_ratio2 = 1
        while not game[3]:
            move = None
            if not game[1]:
                #player 0
                root1 = MCTS(model1,root1,game,iterations,noise,likelihood_modifier=likelihood_ratio1)
                game_reps.append(representation(game))
                move = max(root1.children,key=lambda x: x.score).move
                game_moves.append(move)
            else:
                root2 = MCTS(model2,root2,game,iterations,noise,likelihood_modifier=likelihood_ratio2)
                game_reps.append(representation(game))
                move = min(root2.children,key=lambda x: x.score).move
                game_moves.append(move)
            make_move(game,move)

            found_c1 = False
            for c1 in root1.children:
                if c1.move == move:
                    likelihood_ratio1 = 1/(root1.eval_score())
                    root1 = c1
                    found_c1 = True
                    break
            if not found_c1:
                likelihood_ratio1 = 1
                root1 = Node(None,game)

            found_c2 = False
            for c2 in root2.children:
                if c2.move == move:
                    likelihood_ratio2 = 1/(root2.eval_score())
                    root2 = c2
                    found_c2 = True
                    break
            if not found_c2:
                likelihood_ratio2 = 1
                root2 = Node(None,game)

        if not game[2]:
            #player 0 wins
            score -= 1
        else:
            score += 1

        game_data.append((game_reps,game_moves,game[2]))
    for i in range(n):
        #new model plays white
        reset(game)
        game_reps = []
        game_moves = []
        root1 = Node(None,game)
        root2 = Node(None,game)
        likelihood_ratio1 = 1
        likelihood_ratio2 = 1
        while not game[3]:
            move = None
            if not game[1]:
                #player 0
                root2 = MCTS(model2,root2,game,iterations,noise,likelihood_modifier=likelihood_ratio2)
                game_reps.append(representation(game))
                move = max(root2.children,key=lambda x: x.score).move
                game_moves.append(move)
            else:
                root1 = MCTS(model1,root1,game,iterations,noise,likelihood_modifier=likelihood_ratio1)
                game_reps.append(representation(game))
                move = min(root1.children,key=lambda x: x.score).move
                game_moves.append(move)
            make_move(game,move)

            found_c1 = False
            for c1 in root1.children:
                if c1.move == move:
                    likelihood_ratio1 = 1/(root1.eval_score())
                    root1 = c1
                    found_c1 = True
                    break
            if not found_c1:
                likelihood_ratio1 = 1
                root1 = Node(None,game)

            found_c2 = False
            for c2 in root2.children:
                if c2.move == move:
                    likelihood_ratio2 = 1/(root2.eval_score())
                    root2 = c2
                    found_c2 = True
                    break
            if not found_c2:
                likelihood_ratio2 = 1
                root2 = Node(None,game)

        if not game[2]:
            #new model wins
            score += 1
        else:
            score -= 1
        game_data.append((game_reps,game_moves,game[2]))
    print("New model score: " + str((2*n+score)/(4*n)))
    return (score > score_threshold), game_data