import time, torch

from domineering_game import *
from zobrist_hashing import *

from collections import OrderedDict
import heapq

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class PUCT_Node:
    _id_counter = 0 #used to break ties in comparisons in heap

    def __init__(self, game, zobrist_hash, move, parent=None, win_prob=None, move_probs=None, children=None):
        self.game = game #stores the game state at this node
        self.move = move #stores the move that led to this node
        self.zobrist_hash = zobrist_hash #stores the zobrist hash of the game state at this node
        self.parent = parent #stores the parent node, if any
        self.id = PUCT_Node._id_counter #stores the id of this node
        PUCT_Node._id_counter += 1
        self.win_prob = win_prob #stores the estimated win probability for the first player at this node
        self.move_probs = move_probs #stores the estimated probabilities of each move from this node, computed by model
        self.visits = 1 #stores the number of times this node has been visited, used for selection in MCTS
        self.done = len(legal_moves(game)) == 0 #stores whether this node can be expanded. Updated during search.
        self.depth = 1+parent.depth if parent else 0 #stores the depth of this node in the tree
        self.children = children if children is not None else []
    
    def add_child(self, child):
        self.children.append(child)

    def count_descendants(self):
        # Count the number of descendants of this node
        count = 1
        for child in self.children:
            count += child.count_descendants()
        return count
    
    def __eq__(self, other):
        # Compare nodes by id
        return self.id == other.id
    
class MCTS_PUCT:
    def __init__(self, game, model, move_hashes, zobrist_hashes, prev_evaluations, C=1.0):
        self.model = model
        self.move_hashes = move_hashes
        self.prev_evaluations = prev_evaluations
        zobrist_hash = compute_hash(game[0], zobrist_hashes)
        self.root = PUCT_Node(game, zobrist_hash, None)
        self.root.win_prob = 0.5
        self.root.move_probs = np.ones(N_MOVES)/N_MOVES
        self.C = C #exploration parameter, higher values favor breadth

    def search(self,root):
        node = self.select()
        if node:
            self.expand(node)
            return True
        return False

    def select(self):
        # Starting from root, successively select child with highest exploit score
        # Exploit score = win_prob (from model) + C*move_prob (from model)*sqrt(log(visits to previous node))/(visits to current node)
        # Ignore nodes that have been fully expanded
        # every time we select a node, we add 1 to its visits
        current = self.root
        if current.done:
            #MCTS is done
            return None
        while current.children:
            #find the child with the highest exploit score which has not been fully expanded
            current = max([child for child in current.children if not child.done], key=lambda child: child.win_prob + self.C*current.move_probs[child.move]*np.sqrt(np.log(current.visits)/child.visits))
        return current
    
    def batch_evaluate(self,nodes):
        # Batch evaluate nodes
        nodes_to_eval = []
        boards_to_eval = []
        for node in nodes:
            if node.game[3]:
                # Game is over
                # Winner is game[2] stored as boolean
                node.win_prob = 1 if node.game[2] else 0
            elif node.zobrist_hash in self.prev_evaluations.cache:
                # Game is not over, but eval is known
                win_prob, move_prob = self.prev_evaluations.get(node.zobrist_hash)
                node.win_prob = win_prob
                node.move_probs = move_prob
            else:
                # Needs to be evaluated by model
                nodes_to_eval.append(node)
                boards_to_eval.append([representation(node.game)]) # Shape: (1,BOARD_SIZE,BOARD_SIZE) bc 1 channel

        if not nodes_to_eval:
            # No nodes to evaluate
            return
        
        boards_to_eval = torch.tensor(boards_to_eval, dtype=torch.float32)
        win_probs, move_probs = self.model.predict(boards_to_eval)
        win_probs = win_probs.detach().numpy()
        move_probs = move_probs.detach().numpy()
        for i, node in enumerate(nodes_to_eval):
            node.win_prob = win_probs[i][0]
            node.move_probs = move_probs[i]
            self.prev_evaluations.put(node.zobrist_hash, (win_probs[i][0], move_probs[i]))

    def expand(self,node):
        # Create children of node and batch evaluate them

        new_nodes = []
        for move in range(N_MOVES):
            if legal_moves(node.game)[move] == 0:
                continue
            new_game = copy_game(node.game)
            make_move(new_game, move)
            new_hash = node.zobrist_hash ^ self.move_hashes[move]
            new_node = PUCT_Node(new_game, new_hash, move, node)
            node.add_child(new_node)
            new_nodes.append(new_node)
        
        # Evaluate the new nodes
        self.batch_evaluate(new_nodes)
        
        # Traveling up the tree, update the win probabilities (with minimax), the finished status, and the number of visits
        currentNode = node
        score_changing = True
        finished_changing = True
        while currentNode:
            #update visits
            currentNode.visits += 1

            #update win_prob
            if score_changing:
                prev_score = currentNode.win_prob
                if not currentNode.game[1]:
                    # Player 0
                    # Maximize win prob
                    currentNode.win_prob = max([child.win_prob for child in currentNode.children])
                else:
                    # Player 1
                    # Minimize win prob
                    currentNode.win_prob = min([child.win_prob for child in currentNode.children])
                score_changing = (prev_score != currentNode.win_prob)
            
            if finished_changing:
                prev_finished = currentNode.done
                currentNode.done = any([child.done for child in currentNode.children])
                finished_changing = (prev_finished != currentNode.done)
            
            currentNode = currentNode.parent
        
    def make_move(self, move):
        # First, if the root has not been expanded, expand it
        if not any(self.root.children):
            self.expand(self.root)
        if not any(self.root.children):
            # Root is terminal
            return
        # Update the root node based on the chosen move
        for child in self.root.children:
            if child.move == move:
                child.parent = None  # Detach the new root from its parent
                self.root = child
                return
        # Move was not legal
        raise ValueError("Illegal move")
    
def self_play(model, move_hashes, zobrist_hashes, prev_evaluations, C=0.1, time_per_move = 1.0):
    # Play a game against itself using MCTS
    game = domineering_game()
    mcts = MCTS_PUCT(game, model, move_hashes, zobrist_hashes, prev_evaluations, C)
    still_searching = True
    while not mcts.root.game[3]:
        # Play until the game is over
        time0 = time.time()
        while (time.time() - time0 < time_per_move) & still_searching:
            # Search until time is up
            # Time is measured in second
            still_searching = mcts.search(mcts.root)
        # Choose the best move
        move = None
        if mcts.root.game[1]:
            # Player 1
            # Minimize win prob
            move = min(mcts.root.children, key=lambda child: child.win_prob).move
        else:
            # Player 0
            # Maximize win prob
            move = max(mcts.root.children, key=lambda child: child.win_prob).move
        mcts.make_move(move)
    return mcts.root.game[5], mcts.root.game[2] # Return the history of moves and the winner

def Arena(model1,model2,move_hashes,zobrist_hashes,prev_evals1,prev_evals2,C=0.1,time_per_move=1.0):
    # Play a game between two models
    game1 = domineering_game()
    game2 = domineering_game()
    mcts1 = MCTS_PUCT(game1, model1, move_hashes, zobrist_hashes, prev_evals1, C)
    mcts2 = MCTS_PUCT(game2, model2, move_hashes, zobrist_hashes, prev_evals2, C)
    still_searching = True
    while not mcts1.root.game[3]:
        # Play until the game is over
        # Alternate players
        time0 = time.time()
        while (time.time() - time0 < time_per_move) & still_searching:
            still_searching = mcts1.search(mcts1.root)
        move = max(mcts1.root.children, key=lambda child: child.win_prob).move
        mcts1.make_move(move)
        mcts2.make_move(move)
        if mcts1.root.game[3]:
            break

        time0 = time.time()
        while (time.time() - time0 < time_per_move) & still_searching:
            still_searching = mcts2.search(mcts2.root)
        move = min(mcts2.root.children, key=lambda child: child.win_prob).move
        mcts1.make_move(move)
        mcts2.make_move(move)
    return mcts1.root.game[5], mcts1.root.game[2] # Return the history of moves and the winner

def human_vs_model(human_turn,model,move_hashes,zobrist_hashes,prev_evaluations,C=0.1,time_per_move=5.0):
    # Play a game between a human and a model
    game = domineering_game()
    mcts = MCTS_PUCT(game, model, move_hashes, zobrist_hashes, prev_evaluations, C)
    still_searching = True
    while not mcts.root.game[3]:
        # Play until the game is over
        if human_turn == mcts.root.game[1]:
            # Human's turn
            display(mcts.root.game)
            legal_moves_list = list(legal_moves(mcts.root.game).nonzero()[0])  # Get a list of legal moves
            valid_move = False
            while not valid_move:
                # Get a valid move
                move = int(input("Enter a move: "))
                if move in legal_moves_list:
                    valid_move = True
                else:
                    print("Invalid move")
            mcts.make_move(move)
        else:
            # Model's turn
            time0 = time.time()
            while (time.time() - time0 < time_per_move) & still_searching:
                still_searching = mcts.search(mcts.root)
            move = None
            if human_turn:
                # In Python, True == 1
                # So computer is player 0, so maximize win prob
                move = max(mcts.root.children, key=lambda child: child.win_prob).move
            else:
                # Computer is player 1, so minimize win prob
                move = min(mcts.root.children, key=lambda child: child.win_prob).move
            mcts.make_move(move)
    display(mcts.root.game)
    if mcts.root.game[2]:
        print("Model wins!")
    else:
        print("Human wins!")