import numpy as np
import copy

#Functional implementation of the combinatorial game Domineering
#First player is vertical, second player is horizontal
#Players alternate placing nonoverlapping dominoes on the board
#A player loses if they cannot make a move

#setting constants

BOARD_SIZE = 8 #size of board, can be changed to any integer > 1
N_MOVES = int(2*BOARD_SIZE*(BOARD_SIZE-1)) #number of valid moves on that board
P_MOVES = int(N_MOVES/2) #number of valid moves for one player

ones_array = np.ones(P_MOVES,dtype=bool)
zeros_array = np.zeros(P_MOVES,dtype=bool)

player_1_legal_moves = np.concatenate([ones_array,zeros_array])
player_2_legal_moves = np.concatenate([zeros_array,ones_array])

#pre-calculate moves eliminated by each possible move

ones_array = np.ones(P_MOVES,dtype=bool)
zeros_array = np.zeros(P_MOVES,dtype=bool)

player_1_legal_moves = np.concatenate([ones_array,zeros_array])
player_2_legal_moves = np.concatenate([zeros_array,ones_array])

move_eliminations = []

for move in range(0,N_MOVES):
    valid_moves = np.ones(N_MOVES, dtype=bool)

    if move<P_MOVES:
        i,j = divmod(move,BOARD_SIZE)
        equivalent_move = BOARD_SIZE*j + i + P_MOVES
        p1_eliminated_moves = [move-8,move,move+8]
        p2_eliminated_moves = [equivalent_move - BOARD_SIZE, equivalent_move - BOARD_SIZE+1, equivalent_move, equivalent_move+1]
        for m in p1_eliminated_moves:
            if 0 <= m < P_MOVES:
                valid_moves[m] = False
        for m in p2_eliminated_moves:
            if P_MOVES <= m < N_MOVES:
                valid_moves[m] = False
    else:
        j,i = divmod(move-P_MOVES,BOARD_SIZE)
        equivalent_move = BOARD_SIZE*i + j
        p1_eliminated_moves = [equivalent_move - BOARD_SIZE, equivalent_move - BOARD_SIZE+1, equivalent_move, equivalent_move+1]
        p2_eliminated_moves = [move-8,move,move+8]
        for m in p1_eliminated_moves:
            if 0 <= m < P_MOVES:
                valid_moves[m] = False
        for m in p2_eliminated_moves:
            if P_MOVES <= m < N_MOVES:
                valid_moves[m] = False

    move_eliminations.append(valid_moves)

#Implementing the game

def domineering_game():
    #returns a list of all the data required for a Domineering game
    state = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=bool)
    player = False #player 0, boolean
    winner = None
    done = False
    remaining_moves = np.ones((N_MOVES,),dtype=bool)
    history = []
    return [state,player,winner,done,remaining_moves,history]

def reset(game):
    state = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=bool)
    remaining_moves = np.ones(N_MOVES,dtype=bool)
    history = []
    game[0] = state
    game[1] = False
    game[2] = None
    game[3] = False
    game[4] = remaining_moves
    game[5] = history

def representation(game):
    return np.copy(game[0])

def legal_moves(game):
    # return a list of legal moves for the current player
    if not game[1]:  # If current player is False (player 0)
        return game[4] & player_1_legal_moves
    else:  # If current player is True (player 1)
        return game[4] & player_2_legal_moves

def make_move(game, move):
    # Check if the move is legal
    if not legal_moves(game)[move]:
        return False

    i = None
    j = None

    if not game[1]:
        #game[1] = False = 0, so vertical player
        i,j = divmod(move,BOARD_SIZE)
    else:
        #horizontal player
        j,i = divmod(move-P_MOVES,BOARD_SIZE)

    #place the dommino
    if not game[1]:  # Player 0 (vertical)
        game[0][i, j] = True
        game[0][i+1, j] = True
    else:  # Player 1 (horizontal)
        game[0][i, j] = True
        game[0][i, j+1] = True

    #eliminate moves
    game[4] = game[4] & move_eliminations[move]

    # Append move to history
    game[5].append(move)

    # Switch the player
    game[1] = not game[1]

    # Check if the game is over
    if not any(legal_moves(game)):
        game[2] = not game[1]  # Set the winner to the player before the last move, so True if first and False if second
        game[3] = True #game is done

    return True

def copy_game(game):
    state_copy = np.copy(game[0])
    player_copy = game[1]  # Direct copy is fine for a boolean
    winner_copy = game[2]  # Direct copy is fine for a simple data type like None or boolean
    done_copy = game[3] #Direct copy is fine for a boolean
    remaining_moves_copy = np.copy(game[4])
    history_copy = game[5][:]  # Shallow copy of the list is enough if it contains only primitives
    return [state_copy, player_copy, winner_copy, done_copy, remaining_moves_copy, history_copy]

def display(game):
    state, player, winner, _, _, _ = game
    header_row = ' '
    for i in range(BOARD_SIZE):
        header_row = header_row + " " + str(i%10)
    print(header_row)
    for i in range(BOARD_SIZE):
        print(f'{i%10} ', end='')
        for j in range(BOARD_SIZE):
            if state[i, j]:
                print('X', end=' ')
            else:
                print('_', end=' ')
        print()
    print(f"Player to move: {'0' if not player else '1'}")
    if winner is not None:
        print(f"Winner: {'0' if not winner else '1'}")
    else:
        print("No winner yet")