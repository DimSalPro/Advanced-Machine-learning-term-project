
# Project Title

A project for the Advanced Machine Learning course. It implements the Adam Optimizer optimizer for training feed-forward neural 
networks using the Back-Propagation method for automatically computing the gradient of the loss function of the neural network 
for regression.

## Packages

Required packages: numpy, matplotlib

```bash
  pip install numpy
  pip install matplotlib
```
    
## Configuration

The config2.json file has the parameters that gave the best accuracy score for the dataset provided (OPTION2_train_data.csv).

The config.json file contains all the different parameters that the algorithm can take. You can change it to your own liking.
Speciffically:

    -Hidden Layers: list. The number of items in it corresponds to the number of hidden layers that will be used and the 
    number of each item corresponds to the nodes of each layer.

    -Hidden Layers Activation functions: list. Must have equal size with the Hidden Layers. Each item corresponds to the 
    activation function of each layer of the Hidden Layers list. Available: relu, leakyRelu, tangent, sigmoid.

    -Output Layer Activation Function: string. The activation function of the output node from the available choices(above).
    
    -epochs: integer. Number of epochs.

    -batch size: integer. Number of batches to split the training data.

    -learning rate: float. The number for the learning rate.

    -beta 1: float. The beta 1 parameter is the exponential decay rate of the 1st moment of the Adam Optimizer.

    -beta 2: float. The beta 2 parameter is the exponential decay rate of the 2nd moment of the Adam Optimizer.

    -epsilon: float. Is a small number added to the update of weights to avoid division by zero.

    -k-fold cross validation: integer. Is the number of folds to split the traing set

    -compute_cost_acc_every_x_epochs: integer. Every when to compute loss and accuracy.

    -change_validation_set_every_x_epochs: integer. Every when to move to the next validation fold.

    -weights_init_Gaussian_mean: float. The mean of the Gaussian for the random initialization of the weights.

    -weights_init_Gaussian_standard_deviation: float. The standard deviation of the Gaussian for the random initialization 
    of the weights.

    -loss_criterion: string. The cost function to be used. Available: mse (mean squared error), msle (mean squared logarithmic error)

    -train data directory: string. The directory where the training set resides.

    -test data directory: string. The directory where the testing set resides (if any). Leave it as " " if no testing set
    is available.

    -test_data: string. Set to yes if there is a testing set and you want to have predictions for it or set to no if there
    is no available test set.

    -separator: string. Contains the separator ot the columns of the train and test set. Available: " ", "," etc.

    -headers: string. Set to true if the train and test set have headers row or set to false if not.

    -normalize: string. Set to true if you want the data to be normilized before the training and testing starts or set 
    to false if you want to process the original values.




## Deployment

To run this project, the Project_Final.py and config.json files must be in the same directory.

```bash
  python Project_Final.py
```

