import torch
import numpy as np
from domineering_game import *
from zobrist_hashing import *
from toy_models import *
from domineering_model import *
from MCTS import *

def load_game_history(data_path):
    #A game history is a text file with a game on each line
    #Each line is a list of moves and a result
    try:
        game_history = []
        with open(data_path, 'r') as f:
            for line in f:
                game = line.split(",")
                moves = [int(move) for move in game[:-1]]
                result = int(game[-1])
                game_history.append([moves,result])
        return game_history
    except:
        print("No game history found at ",data_path)
        return []

def prepare_data(game_data):
    #game_data is a double array
    #first element is a list of moves
    #second element is the result
    states = []
    moves = []
    results = []
    for game in game_data:
        gamestate = domineering_game()
        for move in game[0]:
            states.append([gamestate[0]])
            move_array = np.zeros(N_MOVES)
            move_array[move] = 1
            moves.append(move_array)
            results.append(1 - int(game[1]))
            make_move(gamestate,move)
    return [states,moves,results]

def self_play_training(model_path,data_path,self_play_time,arena_time,move_hashes,zobrist_hashes,prev_eval_size,epochs,games_per_epoch,arena_games_per_epoch,noise=0.03):
    #initialize the model
    model = DomineeringModel()
    if model_path:
        #Check if the file exists. If it does, load the model
        try:
            model.load(model_path)
        except:
            print("No model found at ",model_path)
    
    #initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    game_history = []
    if data_path:
        game_history = load_game_history(data_path)
    new_game_history = []

    for epoch in range(epochs):
        prev_evaluations = LRUCache(prev_eval_size)
        print("Epoch ",epoch)
        for game in range(games_per_epoch):
            # Generate a self-play game
            try:
                game_data = self_play(model,move_hashes,zobrist_hashes,prev_evaluations,noise=noise,time_per_move=self_play_time,verbose=False)
                game_history.append(game_data)
                new_game_history.append(game_data)
            except:
                print("Game ",game+1," failed")
        # Save the game history
        with open(data_path, 'a+') as f:
            for game in new_game_history:
                line = ""
                for move in game[0]:
                    line += str(move) + ","
                line += str(int(game[1]))
                f.write(line + "\n")

        new_game_history = []

        # Now, we want to train the model on the game history
        # We'll keep track of the previous version of the model for comparison
        # Train over 10 epochs with a small batch size
        prev_model = model.copy()
        train_data = prepare_data(game_history)
        loss = model.train(optimizer,train_data)
        prev_evaluations2 = LRUCache(prev_eval_size)

        # Now, we want to play a few games against the previous version of the model
        arena_score = 0
        for game in range(arena_games_per_epoch):
            try:
                game_data = Arena(model,prev_model,move_hashes,zobrist_hashes,prev_evaluations,prev_evaluations2,noise=noise,time_per_move=arena_time,verbose=False)
                if game_data[1] == False:
                    # Model 1 won
                    arena_score += 1
                game_history.append(game_data)
                new_game_history.append(game_data)
            except:
                print("Arena game ",2*game+1," failed")

            try:
                # Play a game with the other model going first
                game_data = Arena(prev_model,model,move_hashes,zobrist_hashes,prev_evaluations2,prev_evaluations,noise=noise,time_per_move=arena_time,verbose=False)
                if game_data[1] == True:
                    # Model 2 won
                    arena_score += 1
                game_history.append(game_data)
                new_game_history.append(game_data)
            except:
                print("Arena game ",2*game+2," failed")
        
        print("Arena score: ", arena_score*100/(2*arena_games_per_epoch) , "%")
        if arena_score > arena_games_per_epoch:
            print("New best model! Saving...")
            model.save(model_path)
        else:
            print("Old model is still the best")
            model = prev_model

        # Save the game history
        with open(data_path, 'a+') as f:
            for game in new_game_history:
                line = ""
                for move in game[0]:
                    line += str(move) + ","
                line += str(int(game[1]))
                f.write(line + "\n")
        
        new_game_history = []

    return model

if __name__ == "__main__":
    #initialize the zobrist hashes
    move_hashes = np.load("zobrist_hashes/zobrist_hashes_8x8.npy")
    zobrist_hashes = np.load("zobrist_hashes/zobrist_hashes_8x8_zobrist.npy")
    prev_eval_size = 10000000

    self_play_training("models/domineering_8x8_model.pth","training_data/domineering_8x8_data.npy",0.1,0.2,move_hashes,zobrist_hashes,prev_eval_size,10,100,50)