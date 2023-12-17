Python code for the AISTATS submission of the manuscript "Graph Structure Inference with BAM: Introducing the Bilinear Attention Mechanism"

To run the code:
Install dependencies: python 3.10.8, Tensorflow, Numpy, Pandas
Using conda, you can use conda env create -f env_BAM.yml

To train the neural network, use python edge_classifier_BAM.py

Hyperparameters of the model can be changed in edge_classifier_BAM.py by changing
model = cl.model_attention_final(n_channels_main=100, data_layers=10, cov_layers=10, inner_channels=100, N_exp=3, N_heads=5)

And for changing the number of samples / number of nodes / epochs for training, do changes in edge_classifier_BAM.py here:

spe = 128
ep = 1000
N = 1
M_min = 50
M_max = 1000
d_min = 10
d_max = 100

