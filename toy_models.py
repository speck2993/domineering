import math
import numpy as np
import torch

from domineering_game import *
from zobrist_hashing import *

class ToyModel1:
    def __init__(self):
        pass

    def predict(self,representations):
        #representations is a torch tensor of shape (batch_size,1,BOARD_SIZE,BOARD_SIZE)
        prediction_scores = []
        prediction_likelihoods = []
        for r in representations:
            score = np.random.rand(1)
            move_evals = np.random.rand(N_MOVES)
            prediction_scores.append(score)
            prediction_likelihoods.append(move_evals)
        return [torch.tensor(np.array(prediction_scores)),torch.tensor(np.array(prediction_likelihoods))]

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

    return sigmoid(score/3)

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
        return [torch.tensor(np.array(prediction_scores)),torch.tensor(np.array(prediction_likelihoods))]