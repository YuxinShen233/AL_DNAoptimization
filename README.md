# Optimization of regulatory DNA with active learning

Code and data for paper [“Optimization of regulatory DNA with active learning”](https://www.biorxiv.org/content/10.1101/2025.06.27.661924v1) by Shen, Kudla and Oyarzún.

data.zip - includes all NK landscapes in csv format.
 
code.zip - includes Python code for reproducing the results of the paper.

### 1. Code overview.
It contains two subfolders on NK landscape and promoter landscape respectively, and one environment file.
- `AL.yml`: the environment for all the code

## AL on NK landscape 
### 2. NK genotype-phenotype landscapes (Figure 1)
- `nk_landscape.ipynb`: Generate the NK0-NK3 landscapes and save them in csv files as ground truth landscapes. The NK model is derived from a previous NK simulation in paper [1] from https://github.com/acmater/NK_Benchmarking/blob/master/utils/nk_utils/NK_landscape.py. 
- `nk_local_landscape.ipynb`: Generate the NK1-NK3 local landscapes. 
- `nk_tsne.ipynb`: Plot the 2D t-SNE embedding plots of the genotype space, and label the seqeunces according to their phenotype (Figure 1C).
- `nk_mlp.ipynb`: Train MLP models on four NK landscapes (Figure 1D).

### 3. AL on NK genotype-phenotype landscapes (Figure 2)
- `AL_NK_pipeline.ipynb`: The active learning pipeline on NK landscape. Different conditions like AL with random sampling and ALDE can be set inthe pipeline. The main UCB reward function is adapted from the paper [3].
- `NK_benchmarking_ho.ipynb`: One-shot model performance on the NK landscapes with hyperparameter optimization to compare with AL performance. Three optimization methods on one-shot modelling are implemented: random screening (RS), strong-selection weak-mutation (SSWM) and gradient descent (GD).

## AL on Promoter landscape
### 4. AL on NK genotype-phenotype landscapes (Figure 3)
- `Glu_model.py`, `Ura_model.py`: The code to use the pre-trained promoter landscape. The promoter landscape is derived from the trained transformer structure with a large-scale characterization of promoter expression in paper [2] from https://github.com/1edv/evolution/.
- `AL_loop.py`: The main script for active learning pipeline on promoter landscape.
- `AL_sampling_methods.py`: The selection methods for the active learning pipeline on promoter landscape.
- `AL_selection.py`: The UCB function for the active learning pipeline on promoter landscape, adapted from the paper [3].
- `promoter_benchmarking_ho.ipynb`:  One-shot model performance on promoter landscape with hyperparameter optimization to compare with AL performance. Three optimization methods on one-shot modelling are implemented: random screening (RS), strong-selection weak-mutation (SSWM) and gradient descent (GD).
  
### 5. Biological sampling and motif information (Figure 4)
- `motif_analysis.ipynb`: Conduct motif analysis for the batches sampled by AL. (Figure 4C)
- `AL_PFM.py`: Combine the motif information calculation into the UCB function.



## Reference
[1] Mater, Adam C., Mahakaran Sandhu, and Colin Jackson. "The nk landscape as a versatile benchmark for machine learning driven protein engineering." bioRxiv (2020): 2020-09.

[2] Vaishnav, Eeshit Dhaval, et al. "The evolution, evolvability and engineering of gene regulatory DNA." Nature 603.7901 (2022): 455-463.

[3] Borkowski, Olivier, et al. "Large scale active-learning-guided exploration for in vitro protein production optimization." Nature communications 11.1 (2020): 1872.
