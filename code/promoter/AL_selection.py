import warnings
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
from AL_PFM_and_penalization import *
from AL_sampling_methods import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def select_current_best_model(X, y, Xplot, yplot, models_number = 10, 
                              verbose = False, 
                             MLP = True,
                             visu = False,
                             model_name = "test"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    trained_model_list = []
    # Training all models
    for i in range(models_number):
        X_train, y_train = X, y
        if MLP:
            other_MLP = MLPRegressor(hidden_layer_sizes  = (10, 100,100, 20), solver ="adam", max_iter=20000, 
                                      early_stopping = True, learning_rate = "adaptive")
            other_MLP.fit(X_train, y_train.flatten())    
            trained_model_list.append(other_MLP)

            big_MLP = MLPRegressor(hidden_layer_sizes  = (100,100, 20),solver ="adam", max_iter=20000, 
                                      early_stopping = True, learning_rate = "adaptive")
            big_MLP.fit(X_train, y_train.flatten())    
            trained_model_list.append(big_MLP)


            medium_MLP = MLPRegressor(hidden_layer_sizes  = (40, 10), solver ="adam", max_iter=20000, 
                                      early_stopping = True, learning_rate = "adaptive")
            medium_MLP.fit(X_train, y_train.flatten())    
            trained_model_list.append(medium_MLP)
  
            small_MLP = MLPRegressor(hidden_layer_sizes  = (10), solver ="adam", max_iter=20000, 
                                      early_stopping = True, learning_rate = "adaptive")
            small_MLP.fit(X_train, y_train.flatten())    
            trained_model_list.append(small_MLP)
        
    # Evaluating all 
    all_scores = []
    for i in range(len(trained_model_list)):
        selected_mdoel = trained_model_list[i]
        y_pred = selected_mdoel.predict(X)
        score = sklearn.metrics.r2_score(y, y_pred)
        all_scores.append(score)

    try:
        best_index = all_scores.index(max(all_scores))
        best_score = all_scores[best_index]
    except ValueError:
        best_index = 0
    if verbose:
        print(all_scores)
        print("Best index is {}".format(best_index))
        print("Best score is {}".format(best_score))
    best_model = trained_model_list[best_index]
    if visu:        
        model = best_model
        y_pred = model.predict(Xplot)
        score = sklearn.metrics.r2_score(yplot, y_pred)
        fig, ax = plt.subplots()
        ax.scatter(yplot, y_pred, edgecolors=(0, 0, 0))
        ax.plot([0, 20], [0, 20], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_title("Model prediction for model {}: {}".format(model_name, score))
        ax.set_ylabel('Predicted')
        plt.show()
    return(best_model, best_score)

def generate_random_grid(sample_size = 10000, normalisation = True, verbose = True):
    """
    Generates a random grid of desired size avoiding predefined concentrations.
    Can be quite long for big arrays, as it verifies combinations were not previously sampled.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    DNA_random=[]
    for i in range(sample_size):
        DNA_random.append(''.join(random.choice(nucleotides) for _ in range(80)))
    return(DNA_random)

def select_best_predictions_from_ensemble_model(ensemble_of_models, 
                                                sampled_sequences,
                                                sampled_expression,
                                                ensemble_model,
                                                total_sampling_size = 10000,
                                                sample_size = 100,
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
    
#     # ADDING motif information
#     k=2/3
#     y_pred += k * sum_pocc_sequence(active_learning_array)
    
    
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

def build_cnn(input_length):
    model = Sequential()

    # Convolutional blocks
    for _ in range(3):
        model.add(Conv1D(256, 13, activation='relu', input_shape=(input_length, 4),padding='same'))
        model.add(Dropout(0.15))
        model.add(MaxPooling1D(pool_size=2))
        input_length = None  # Only set input_shape for the first layer
    model.add(Flatten())

    # Dense blocks
    for _ in range(4):
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))

    # Final Dense output
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def select_current_best_model_CNN(X, y, Xplot, yplot, models_number=10, 
                                  verbose=False, visu=False, model_name="test"):
    X_cnn = X.reshape(-1, 80, 4)
    trained_model_list = []
    input_length = X_cnn.shape[1]
    for i in range(models_number):
        cnn = build_cnn(input_length)
        cnn.fit(X_cnn, y, epochs=20, batch_size=32, verbose=0)
        trained_model_list.append(cnn)
    # Evaluate all
    all_scores = []
    for model in trained_model_list:
        y_pred = model.predict(X_cnn).flatten()
        score = sklearn.metrics.r2_score(y, y_pred)
        all_scores.append(score)
    best_index = np.argmax(all_scores)
    best_score = all_scores[best_index]
    best_model = trained_model_list[best_index]
    # if verbose:
    #     print(all_scores)
    #     print("Best index is {}".format(best_index))
    #     print("Best score is {}".format(best_score))
    # if visu:
    #     y_pred = best_model.predict(Xplot).flatten()
    #     score = sklearn.metrics.r2_score(yplot, y_pred)
    #     fig, ax = plt.subplots()
    #     ax.scatter(yplot, y_pred, edgecolors=(0, 0, 0))
    #     ax.plot([0, 20], [0, 20], 'k--', lw=4)
    #     ax.set_xlabel('Measured')
    #     ax.set_title("Model prediction for model {}: {}".format(model_name, score))
    #     ax.set_ylabel('Predicted')
    #     plt.show()
    return best_model, best_score

def select_best_predictions_from_ensemble_model_CNN(ensemble_of_models, 
                                                sampled_sequences,
                                                sampled_expression,
                                                ensemble_model,
                                                total_sampling_size = 10000,
                                                sample_size = 100,
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
    exploitation: weighting of the yield for UCT equivalent
    exploration: weighting of the ucnertainty (yield std) for UCT equivalent 
    normalisation: normalise concentrations (maximum at 1) 
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
        y_pred = np.array(model.predict(np.array(seq_to_oh(active_learning_array)).reshape(-1, 80, 4)))
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
    
#     # ADDING motif information
#     k=2/3
#     y_pred += k * sum_pocc_sequence(active_learning_array)
    
    
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