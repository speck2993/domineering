'''
By: Joshua Speckman
Generates 64-bit integers for use in Zobrist hashing for k by k Domineering
Relevant: each position is only reachable by one player, so don't track active player
'''

import numpy as np
import random

def generate_zobrist_hashes(BOARD_SIZE, dest):
    # First, generate a random 64-bit integer for each square
    # We'll use a 2D array to store the integers
    zobrist_hashes = np.random.randint(0, 2**64, (BOARD_SIZE, BOARD_SIZE), dtype=np.uint64)

    # Now, we'll compute the hashes for each possible move
    N_MOVES = 2 * BOARD_SIZE * (BOARD_SIZE - 1)
    P_MOVES = int(N_MOVES / 2)

    move_hashes = np.zeros((N_MOVES,), dtype=np.uint64)
    for m in range(P_MOVES):
        #player 1: vertical
        i,j = divmod(m,BOARD_SIZE)
        hash = np.uint64(0)
        hash ^= zobrist_hashes[i,j]
        hash ^= zobrist_hashes[i+1,j]
        move_hashes[m] = hash
    for m in range(P_MOVES, N_MOVES):
        #player 2: horizontal
        j,i = divmod(m-P_MOVES,BOARD_SIZE)
        hash = np.uint64(0)
        hash ^= zobrist_hashes[i,j]
        hash ^= zobrist_hashes[i,j+1]
        move_hashes[m] = hash
    
    # Save both the hashes and the zobrist table
    np.save(dest + ".npy", move_hashes)
    np.save(dest + "_zobrist.npy", zobrist_hashes)
    return

def load_zobrist_hashes(BOARD_SIZE, src):
    # Load the hashes and zobrist table
    move_hashes = np.load(src + ".npy")
    zobrist_hashes = np.load(src + "_zobrist.npy")
    return move_hashes, zobrist_hashes

def compute_hash(position, zobrist_hashes):
    # Compute the hash of a given position
    # We'll use the bitwise XOR operator (^) to combine the hashes of each square
    hash = np.uint64(0)
    for i in range(len(position)):
        for j in range(len(position[i])):
            if position[i][j] == 1:
                hash ^= zobrist_hashes[i][j]
    return hash

if __name__ == "__main__":
    generate_zobrist_hashes(16, "zobrist_hashes_16x16")
    