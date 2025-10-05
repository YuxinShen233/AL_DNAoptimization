# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import linear_model
from sklearn import decomposition
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from smt.sampling_methods import LHS

import time
import datetime
import random
import csv
import os
import copy
import json

# %%
import tensorflow as tf

def latin_sampling(num):
    xlimits = np.array([[0,1.0]]*80)
    sampling = LHS(xlimits=xlimits,)
    x = sampling(num)
    sequence_list=[]
    for i in range(num):
        oh=""
        for j in range(80):
            if x[i][j]<0.25:
                oh=oh+"A"
            elif x[i][j]<0.5:
                oh=oh+"C"
            elif x[i][j]<0.75:
                oh=oh+"G"
            elif x[i][j]<1:
                oh=oh+"T"             
        sequence_list.append(oh)
    return sequence_list

# %%
def random_sampling(num):
    x= np.random.random_sample((num,80))
    sequence_list=[]
    for i in range(num):
        oh=""
        for j in range(80):
            if x[i][j]<0.25:
                oh=oh+"A"
            elif x[i][j]<0.5:
                oh=oh+"C"
            elif x[i][j]<0.75:
                oh=oh+"G"
            elif x[i][j]<1:
                oh=oh+"T"             
        sequence_list.append(oh)
    return sequence_list

# %%
def oh_to_seq(onehot):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence=[]
    for i in range(len(onehot)):
        decoded_sequence=[]
        for j in range(80):
            decoded_sequence.append(mapping[int(onehot[i][4*j+1]*1+onehot[i][4*j+2]*2+onehot[i][4*j+3]*3)])
        sequence.append(''.join(decoded_sequence))
    return sequence

# %%
def seq_to_oh(sequence):
    onehot=[]
    diction =[['A'], ['C'], ['G'], ['T']]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(diction)
    for i in range(len(sequence)): 
        sep_seq=list(sequence[i])
        sep_seq=np.array(sep_seq)
        sep_seq=sep_seq.reshape(len(sequence[i]),1)
        A=enc.transform(sep_seq).toarray()
        A=A.reshape(len(sequence[i])*4)
        onehot.append(A)
    onehot=np.array(onehot)
    onehot = onehot.astype('float32')
    return onehot

# %%
# all_sample_X = latin_sampling(100000)
# all_sample_y = Ura_surrogate(all_sample_X)

# top_indices_and_values = sorted(enumerate(all_sample_y), key=lambda x: x[1], reverse=True)[-1000:]

# # Extract only the indices
# top_indices = [index for index, _ in top_indices_and_values]
# extracted_elements = [all_sample_X[index] for index in top_indices]
# with open('initial_condition/initial_latin_low_2000.txt', 'w') as out_file:
#     json.dump(extracted_elements, out_file)

# %%
# extracted_elements = latin_sampling(2000)
# with open('initial_condition/inital_latin_uniform_2000.txt', 'w') as out_file:
#     json.dump(extracted_elements, out_file)



# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping




# %% [markdown]
# ### Bio-sampling methods

# %%
def generate_single_mutations(dna_sequence):
    """
    generate all the single mutation neighbours for a given sequence
    """
    mutation_neighbors = []

    for i in range(len(dna_sequence)):
        for base in ['A', 'T', 'G', 'C']:
            if dna_sequence[i] != base:
                mutated_sequence = list(dna_sequence)
                mutated_sequence[i] = base
                mutation_neighbors.append(''.join(mutated_sequence))

    return mutation_neighbors

# %%
def single_mutant_walking(original_sequences, num_mutations=10, sample_size=10000):
    mutated_sequences = []


    for _ in range(sample_size):      # Randomly select one of the original sequences
        original_seq = random.choice(original_sequences)
        mutated_sequence = list(original_seq)

        # Generate 10 random positions for mutations
        mutation_positions = random.sample(range(len(mutated_sequence)), num_mutations)

        for position in mutation_positions:
            # Replace the base at the randomly selected position with a random base
            new_base = random.choice(['A', 'T', 'G', 'C'])
            mutated_sequence[position] = new_base

        mutated_sequences.append(''.join(mutated_sequence))

    return mutated_sequences


# %%
def genetic_drift(original_sequences, mutation_probability=0.1, num_sequences=10000):
    mutated_sequences = []

    for _ in range(num_sequences):
        mutated_sequence = ""

        for i in range(len(original_sequences[0])):
            if random.random() < mutation_probability:
                # Mutate the base at the current position
                new_base = random.choice(['A', 'T', 'G', 'C'])
                mutated_sequence += new_base
            else:
                mutated_sequence += random.choice(original_sequences)[i]

        mutated_sequences.append(mutated_sequence)

    return mutated_sequences

# %%
# from itertools import permutations

# def choose_breaking_sites(sequence_length, num_breaks=10):
#     # Choose num_breaks random breaking sites
#     return sorted(random.sample(range(1, sequence_length), num_breaks))

# def evolution(sequences, num_combinations=10000):
#     breaking_points = choose_breaking_sites(len(sequences[0]))

#     broken_sequences = [[sequence[i:j] for i, j in zip([0] + breaking_points, breaking_points + [None])] for sequence in sequences]
    
#     result_sequences = []

#     all_permutations = list(permutations(range(11)))
#     sampled_permutations = random.sample(all_permutations, 100000)
#     i=0
#     while len(result_sequences) < num_combinations:   
#         seq=[]
#         for j in range(11):
#             seq.append(broken_sequences[int(sampled_permutations[i][j])][j])
#         seq=''.join(map(str, seq))
#         if seq not in result_sequences:
#             result_sequences.append(seq)
#         i=i+1
#     return result_sequences

# %%
from itertools import permutations

def choose_breaking_sites(sequence_length, num_breaks=10):
    # Choose num_breaks random breaking sites
    return sorted(random.sample(range(1, sequence_length), num_breaks))

def evolution(sequences, num_combinations=10000):
    breaking_points = choose_breaking_sites(len(sequences[0]))

    broken_sequences = [[sequence[i:j] for i, j in zip([0] + breaking_points, breaking_points + [None])] for sequence in sequences]
    
    result_sequences = []

    while len(result_sequences) < num_combinations:  
        sampled_permutations = [random.randint(0, 99) for _ in range(11)]
        seq=[]
        for j in range(11):
            seq.append(broken_sequences[int(sampled_permutations[j])][j])
        seq=''.join(map(str, seq))
        if seq not in result_sequences:
            result_sequences.append(seq)
    return result_sequences


