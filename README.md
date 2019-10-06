# sentiment-classification

### Results of different algorithms
| Algorithm   |      Accuracy      |
|----------|:-------------:|
| CNN |  84.59% |
| CNN-SBN-ST |   86.44%   |
| CNN-DBN-ST | 87.54% |

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
    - n_filters [Number of filters/output channels]
    - filter_sizes [A comma separated string of filter sizes, such as "2,3,4,5"]
    - output_dim [The number of outputs, which is 1 in the case of the sentiment classification task]
    - dropout_rate [The probability to which dropout should be applied]
    - binary_neuron [Type of binary neuron to use, has to be D-ST or S-ST]

### Example
    To run the program for 10 epochs with an embeddings dimension of 100 the batch size set to 64 and the D-ST trained Binary Neuron 
    run "python main.py --epochs 10 --embed_dim 100 --batch_size 64 --binary_neuron D-ST
