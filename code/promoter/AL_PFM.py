
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
from AL_sampling_methods import *


all_pfm = pd.read_csv('all_pfm.csv')
all_pfm = list(all_pfm["Motif ID"])
# %%
def calculate_pocc(pfm: pd.DataFrame, sequence: str) -> float:
    # Initialize the product of the probabilities of not binding
    product_not_binding = 1

    # Get the length of the PFM
    pfm_length = len(pfm.columns)

    # Create a mapping from bases to indices
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # For each window in the sequence
    for i in range(len(sequence) - pfm_length + 1):
        # Get the window
        window = sequence[i:i+pfm_length]

        # Calculate the probability of binding to the current window (Pi)
        pi = 1
        for j in range(pfm_length):
            # Get the base at position j in the window
            base = window[j]
            # Get the index of the base
            base_index = base_to_index[base]
            # Get the probability of the base at position j
            prob = pfm.loc[base_index, j] 
            # Multiply Pi by the probability
            pi *= prob

        # Update the product of the probabilities of not binding
        product_not_binding *= (1 - pi)

    # Calculate Pocc
    pocc = 1 - product_not_binding

    # Return Pocc
    return pocc

# %%
def sum_pocc_sequence(sequences):
    pfm_value = np.zeros(len(sequences))
    for i in range(244):
        pfm_file_path = 'PFM/'+all_pfm[i]+'.pfm'
        pfm_df = pd.read_csv(pfm_file_path, delimiter='\t', skiprows=0,header=None)
        for j in range(len(sequences)):
            pfm_value[j]+=calculate_pocc(pfm_df, sequences[j])    
    return pfm_value


def select_best_predictions_pfm(ensemble_of_models, 
                                                sampled_sequences,
                                                sampled_expression,
                                                ensemble_model,
                                                total_sampling_size = 10000,
                                                sample_size = 1000,
                                                exploitation = 0.7,
                                                exploration = 0.3, 
                                               initial_max = 1,
                                               verbose = True,
                                               sampling = "random"): 
    """
    Heart of the active learning process.
    Uses a pre-trained ensemble of models to predict on randomly chosen combinations and selects next experiments.
    total_sampling_size: number of combinations to randomly sample
    sample_size: number to export for further analysis
    exploitation: weighting of the yield 
    exploration: weighting of the ucnertainty (yield std)
    sampling = ["random", "DE", "drift", "recombination"]
    """
    # Random 
    if sampling == "random":
        active_learning_array = generate_random_grid(sample_size = total_sampling_size, normalisation = True)

    # DE 
    if sampling == "DE":
        top_100_indices = [index for index, value in sorted(enumerate(sampled_expression[-1]), key=lambda x: x[1], reverse=True)[:100]]
        selected_elements = [sampled_sequences[-1][i] for i in top_100_indices]
        active_learning_array = single_mutant_walking(selected_elements, sample_size=total_sampling_size)

    # genetic drift
    if sampling == "drift":    
        top_100_indices = [index for index, value in sorted(enumerate(sampled_expression[-1]), key=lambda x: x[1], reverse=True)[:100]]
        selected_elements = [sampled_sequences[-1][i] for i in top_100_indices]
        active_learning_array = genetic_drift(selected_elements)
        
    
    # recombination
    if sampling == "recombination":
        top_100_indices = [index for index, value in sorted(enumerate(sampled_expression[-1]), key=lambda x: x[1], reverse=True)[:100]]
        selected_elements = [sampled_sequences[-1][i] for i in top_100_indices]
        active_learning_array = evolution(selected_elements)
        

    # Predicting the full random grid
    answer_array_pred = np.empty
    all_predictions = None
    if verbose:
        print("Starting ensemble predictions")
    for model in ensemble_of_models:
        y_pred = np.array(model.predict(np.array(seq_to_oh(active_learning_array))))
        print(len(y_pred))
        answer_array_pred = y_pred.reshape(total_sampling_size, -1)
        if all_predictions is None:
            all_predictions = y_pred.reshape(total_sampling_size, -1)
        else:
            all_predictions =np.concatenate((all_predictions, y_pred.reshape(total_sampling_size, -1)), axis = 1)
    if verbose:
        print("Finished ensemble predictions")    
        
        
        
    # Obtaining mean and std for predicted array
    y_pred, y_pred_std = np.mean(all_predictions, axis = 1), np.std(all_predictions, axis = 1)

    
    # ADDING motif information
    k=1/2
    y_pred += k * sum_pocc_sequence(active_learning_array)
    
    
    # Create the array to maximise, balancing between exploration and exploitation
    array_to_maximise = copy.deepcopy(exploitation * y_pred + exploration * y_pred_std)
    z_scorearray = copy.copy(array_to_maximise)
    z_exploration = copy.copy(y_pred_std)
    z_exploitation = copy.copy(y_pred)
    z_scorearray = z_scorearray.astype('float32')
    z_exploration = z_exploration.astype('float32')
    z_exploitation = z_exploitation.astype('float32')
    # Select arrays depending on choice of way to eplore: only uncertainty, only yield, or a mix of both.
    conditions_list_pure_exploitation = []
    for count in range(sample_size):
        i = np.argmax(y_pred)
        conditions_list_pure_exploitation.append(int(i))
        if verbose:
            print("Maximising sample {} is yield: {}, std = {}".format(i, y_pred[i], y_pred_std[i]))
        y_pred[i] = -1
        
    conditions_list_pure_exploration = []
    for count in range(sample_size):
        i = np.argmax(y_pred_std)
        conditions_list_pure_exploration.append(int(i))
        if verbose:
            print("Maximising sample {} is yield: {}, std = {}".format(i, y_pred[i], y_pred_std[i]))
        y_pred_std[i] = -1
    
    conditions_list = []
    for count in range(sample_size):
        i = np.argmax(array_to_maximise)
        conditions_list.append(int(i))
        if verbose:
            print("Maximising sample {} is yield: {}, std = {}".format(i, y_pred[i], y_pred_std[i]))
        array_to_maximise[i] = -1
    else:
        active_learning_array =  active_learning_array
    active_learning_array=np.array(seq_to_oh(active_learning_array))       
    conditions_to_test = active_learning_array[conditions_list,:]
    conditions_to_test_eploration = active_learning_array[conditions_list_pure_exploration,:]
    conditions_to_test_exploitation = active_learning_array[conditions_list_pure_exploitation,:]
    return(conditions_to_test, conditions_to_test_eploration, conditions_to_test_exploitation,z_scorearray,z_exploitation,z_exploration)


