import math, json
import numpy as np
import sys

from domineering_game import *
from zobrist_hashing import *
from MCTS_test import *

class ToyModel1:
    def __init__(self):
        pass

    def predict(self,representations):
        prediction_scores = []
        prediction_likelihoods = []
        for r in representations:
            score = np.random.rand(1)
            move_evals = np.random.rand(N_MOVES)
            prediction_scores.append(score)
            prediction_likelihoods.append(move_evals)
        return prediction_scores,prediction_likelihoods

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def simple_score(rep):
    score = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE-1):
            if not (rep[j][i] or rep[j+1][i]):
                score += 1
            if not (rep[i][j] or rep[i][j+1]):
                score -= 1

    return sigmoid(score)

class ToyModel2:
    def __init__(self):
        pass

    def predict(self,representations):
        prediction_scores = []
        prediction_likelihoods = []
        for r in representations:
            score = simple_score(r[0])
            move_evals = np.random.rand(N_MOVES)
            prediction_scores.append([score])
            prediction_likelihoods.append(move_evals)
        return prediction_scores,prediction_likelihoods
    
if __name__ == "__main__":
    move_hashes = np.load("zobrist_hashes_8x8.npy")
    zobrist_hashes = np.load("zobrist_hashes_8x8_zobrist.npy")
    tm2 = ToyModel2()
    prev_evaluations2 = LRUCache(4000000)
    human_vs_model(True,tm2,move_hashes,zobrist_hashes,prev_evaluations2,time_per_move=10)