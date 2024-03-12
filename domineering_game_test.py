import random, time

from domineering_game import *

def play_random_game(verbose=0):
    # Play moves completely at random
    game = domineering_game()
    while not game[3]:  # Continue playing until we're done
        legal_moves_list = list(legal_moves(game).nonzero()[0])  # Get a list of legal moves
        random_move = random.choice(legal_moves_list)  # Choose a random legal move
        make_move(game, random_move)  # Make the random move
        if verbose>1:
            print("Move: " + str(random_move))
            display(game)  # Display the current game state

    if game[3]:
        if verbose==1:
            display(game)
        if verbose>0:
            print(f"Player {game[2]} wins!")
        return

    return

if __name__ == "__main__":
    # Call the function to play and display a random game
    time0 = time.time()
    N = 5000

    for i in range(N):
        play_random_game()

    print("Average time: " + str((time.time()-time0)/N))

    #play_random_game(verbose=2)