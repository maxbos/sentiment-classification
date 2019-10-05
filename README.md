# sentiment-classification

### Prerequisite
    - Anaconda (python 3.x)

### Instructions to run
    - First setup conda environment: run "conda env create -f sentclass.yml"
    - Activate the environment with "conda activate sentclass"
    - Run the main.py with python main.py --arguments

### Arguments specification
    - epochs [Amount of epochs to train the model]
    - embed_dim [Size of the the embeddings used in the model]
    - batch_size [Size of the batch to use during training]
    - stochastic_neuron [Type of stochastic neuron to use, has to be REINFORCE or STE]

### Example
    To run the program for 10 epochs with an embeddings dimension of 100 the batch size set to 64 and the REINFORCE trained Stochastic Neuron 
    run "python main.py --epochs 10 --embed_dim 100 --batch_size 64 --stochastic_neuron REINFORCE