# %% [markdown]
# # Importing necessary libraries

# %%
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

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


from AL_selection import *



# %%
import tensorflow as tf

# %%
from Glu_model import *
from Ura_model import *


def noisy_surrogate(sequence):
    result=Ura_surrogate(sequence)
    for i in range(len(result)):
        result[i]= np.random.normal(result[i],result[i]*0.05, 1)[0]
    return result

def very_noisy_surrogate(sequence):
    result=Ura_surrogate(sequence)
    for i in range(len(result)):
        result[i]= np.random.normal(result[i],result[i]*0.10, 1)[0]
    return result


data_X=[]
data_ym=[]
data_yp=[]
data_z=[]

# %%
# initial_sampling=latin_sampling(1000)
# with open('initial_sampling.txt', 'w') as out_file:
#     json.dump(initial_sampling, out_file)

# %%
with open('noisy_and_TL/data_X_SMW_LHS_TL_GluOpti.txt', 'r') as f:
    initial_sampling=json.load(f)

initial_sampling=initial_sampling[-1]

# initial_sampling=initial_sampling[:1000]
# %%
#training initial model
training_mlp_start_time = time.time()
ensemble_MLP = []
n = 10
for i in range(n):
    current_best, best_score = select_current_best_model(seq_to_oh(initial_sampling), Glu_surrogate(initial_sampling), seq_to_oh(initial_sampling), Glu_surrogate(initial_sampling), 
                                             models_number = 10, verbose = False,
                                             MLP = True,
                                             visu = False,
                                             model_name = i)
    ensemble_MLP.append(current_best)
training_mlp_end_time = time.time()

# %%
data_X.append(initial_sampling)
data_ym.append(Glu_surrogate(initial_sampling))

# %%
n=10
for j in range(4):
    print("start",j)
    conditions_to_test, conditions_to_test_exploration, conditions_to_test_exploitation, z, z_m, z_std = select_best_predictions_from_ensemble_model(ensemble_of_models = ensemble_MLP, 
                                            sampled_sequences=data_X,
                                            sampled_expression=data_ym, 
                                            ensemble_model=ensemble_MLP,
                                            total_sampling_size = 10000, 
                                            verbose = True,
                                            sample_size = 1000, sampling="DE")
    print("end",j)
    data_z.append(z)
    current=oh_to_seq(conditions_to_test)
    # for epsilon-greedy
#     current=oh_to_seq(np.vstack((conditions_to_test_exploration[0:10],conditions_to_test_exploitation[0:90])))
    data_X.append(current)
    data_ym.append(Ura_surrogate(current))
    y_pred=[]
    for model in ensemble_MLP:
        y_pred.append(np.array(model.predict(seq_to_oh(list(np.concatenate(data_X[:j+2]))))))
    data_yp.append(np.mean(y_pred,axis=0))
    
    ensemble_MLP = []
    for i in tqdm(range(n)):
        current_best, best_score = select_current_best_model(seq_to_oh(np.concatenate(data_X[0:j+2])),np.concatenate(data_ym[0:j+2]), seq_to_oh(data_X[j+1]),data_ym[j+1], 
                                                 models_number = 10, verbose = False,
                                                 MLP = True,
                                                 visu = False,
                                                 model_name = i)
        ensemble_MLP.append(current_best)
    training_mlp_end_time = time.time()

    with open('data_X_SMW_LHS_TL_AfterOpti.txt', 'w') as out_file:
        json.dump(data_X, out_file)


data_yp=[data_yp[i].tolist() for i in range(4)]
with open('data_yp_SMW_LHS_TL_AfterOpti.txt', 'w') as out_file:
    json.dump(data_yp, out_file)

# %%
data_y=[data_ym[i].tolist() for i in range(5)]
with open('data_ym_SMW_LHS_TL_AfterOpti.txt', 'w') as out_file:
    json.dump(data_y, out_file)

