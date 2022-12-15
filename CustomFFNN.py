import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import random

data = json.load(open('config2.json'))
hidden_layers = data["parameters"]["Hidden Layers"]
hidden_layers_activation = data["parameters"]["Hidden Layers Activation Functions"]
output_layer_activation = data["parameters"]["Output Layer Activation Function"]
epochs = data["parameters"]["epochs"]
batch_size = data["parameters"]["batch size"]
learning_rate = data["parameters"]["learning rate"]
beta1 = data["parameters"]["beta 1"]
beta2 = data["parameters"]["beta 2"]
epsilon = data["parameters"]["epsilon"]
metrics_change_time = data["parameters"]["compute_cost_acc_every_x_epochs"]
validation_change_time = data["parameters"]["change_validation_set_every_x_epochs"]
weights_init_Gaussian_mean = data["parameters"]["weights_init_Gaussian_mean"]
weights_init_Gaussian_standard_deviation = data["parameters"]["weights_init_Gaussian_standard_deviation"]
loss_criterion = data["parameters"]["loss_criterion"]
test_data = data["file"]["test_data"]
train_dir = data["file"]["train data directory"]
test_dir = data["file"]["test data directory"]
separator = data["file"]["separator"]
headers = data["file"]["headers"]
normalize = data["file"]["normalize"]
validation = data["parameters"]["k-fold cross validation"]


class NeuralNetworkRegressor():
    '''
    A class that handles the reading of the files, split to train and validation, training and predicting
    '''

    # Initialize the class with the input parameters that the user provided in the config2.json file
    # Make those parameters parts of self to access them from all the functions
    # Initialize also a time step parameter (counts time steps), a dictionary to save the input parameters (provided
    # by the user) and a dictionary for the weights,bias,forward and back propagation
    def __init__(self, hidden_layers_list, hidden_layers_activation_list, output_layer_activation, epochs, train_dir,
                 test_dir, batch_size, learning_rate, beta1, beta2, epsilon, separator, headers, validation, normalize,
                 metrics_change_time, validation_change_time, weights_init_Gaussian_mean,
                 weights_init_Gaussian_standard_deviation, loss_criterion, test_data):
        self.hidden_layers_list: list = hidden_layers_list
        self.hidden_layers_activation_list: list = hidden_layers_activation_list
        self.output_layer_activation = output_layer_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.headers = headers
        self.separator = separator
        self.validation = validation
        self.metrics_change_time = metrics_change_time
        self.validation_change_time = validation_change_time
        self.Gaussian_mean = weights_init_Gaussian_mean
        self.Gaussian_standard_deviation = weights_init_Gaussian_standard_deviation
        self.loss_criterion = loss_criterion
        self.test_data = test_data
        self.time = 0
        self.normalize = normalize
        self.normalize_parameter = 0
        self.input_parameters_dict = {}
        self.weights_dict = {}
        print('----------------------------IMPORTANT SUGGESTION----------------------------')
        print('If the validation accuracy starts below 52% on the first epoch, It is advised to restart the script.')
        print('Sometimes due to the randomness of the weights initialization (maybe initial weights starts to high)\nand cause the accuracy to stay low!')
        print('----------------------------------------------------------------------------\n')

    # Fills the input parameters dictionary with information about the hidden layers and their nodes
    # and maps the activation functions of each layer from the ActivationFunctions class
    def prepare_input_parameters(self):
        for number_of_layer in range(len(self.hidden_layers_list) + 1):
            if number_of_layer == len(self.hidden_layers_list):
                self.input_parameters_dict['layer ' + str(number_of_layer)] = {
                    'activation': self.output_layer_activation
                }
            else:
                self.input_parameters_dict['layer ' + str(number_of_layer)] = {
                    'activation': self.hidden_layers_activation_list[number_of_layer],
                    'nodes': self.hidden_layers_list[number_of_layer]
                }

        self.input_parameters_dict = ActivationFunctions().mapper(self.input_parameters_dict)

    # takes as input the data, shuffles them, makes them as numpy array and splits them into array of arrays based
    # on self.validation parameter which represents the folds for the cross validation
    # Ex. For a 10-fold cross validation it outputs an array with 10 arrays of equal size in it
    def validation_split(self, data):
        random.shuffle(data)
        data_out = np.asarray(data)
        data_out = np.array_split(data_out, self.validation)

        return data_out

    # When called, it reads the train and test data from the directories provided by the user.
    # If the data have headers it reads the file from the 1st row and afterwards.
    # Preprocess every row to remove the \n on the end, split on the separator provided by the user
    # and maps each string to float so it can be ready for the next steps.
    # Splits dataset into validation folds. Finally it fills the input list with the input nodes from the data given
    # and normalizes the test data based on the maximum value found in the train set
    def read_data_and_map(self):
        dataset_list = []
        with open(self.train_dir, "r") as train_csv:
            train_data = train_csv.readlines()

            if self.headers.lower() == "true":
                train_data = train_data[1:]

            for row in train_data:
                row = row.rstrip("\n").split(separator)
                ds = list(map(float, row))
                max_number = max(ds)
                if max_number > self.normalize_parameter:
                    self.normalize_parameter = max_number
                dataset_list.append(ds)

            dataset_list = self.validation_split(dataset_list)

            output_layer_nodes = 1  # regression
            input_layer_nodes = len(dataset_list[0][0]) - 1  # -1 for the label class
            number_of_output_layer = len(self.hidden_layers_list)
            self.input_parameters_dict['layer ' + str(number_of_output_layer)]['nodes'] = output_layer_nodes
            self.input_parameters_dict['input layer'] = {'nodes': input_layer_nodes}
            self.hidden_layers_list.insert(0, input_layer_nodes) #insert input layer size on the index 0 of the list
            self.hidden_layers_list.append(output_layer_nodes)
            # print(list(self.input_parameters_dict.keys()))
            # print(self.hidden_layers_list)

        x_test = []
        if self.test_data.lower() == "yes":
            with open(self.test_dir, "r") as test_csv:
                test_data = test_csv.readlines()

                if self.headers.lower() == "true":
                    test_data = test_data[1:]

                for row in test_data:
                    row = row.rstrip("\n").split(separator)
                    ds = list(map(float, row))
                    x_test.append(ds)

                if self.normalize.lower() == "true":
                    x_test = np.asarray(x_test) / self.normalize_parameter

        return dataset_list, x_test

    # When called it initializes the weights and biases for all the layers (random number following Gaussian between
    # with configured mean and standard deviation)
    # It constructs the self.weights_dict with the layers weights and biases that will be used to train the model
    def initialize_weights(self):
        input_nodes = self.hidden_layers_list[0]
        count = 0
        previous_node_size = input_nodes

        for layer_node_size in self.hidden_layers_list[1:]:
            current_node_size = layer_node_size
            bias = np.zeros((current_node_size, 1))

            weights = np.random.RandomState(None).normal(loc=self.Gaussian_mean, scale=self.Gaussian_standard_deviation,
                                                         size=(previous_node_size, current_node_size))
            hidden_layer_number = 'layer ' + str(count)

            self.weights_dict[hidden_layer_number] = {'weights': weights, 'bias': bias}

            # print(f'{hidden_layer_number}::: No rows {len(weights)} No cols {len(weights[0])}')
            previous_node_size = current_node_size
            count += 1

    # When called it changes to the new validation and train set. For a 10-fold cross validation it goes to 0,...,9
    # and the 10 % 10 also goes to 0 (starting from the beginning). Then the train set is split into batches
    # (list of lists in which each lists has batch size number of elements). Also it normalizes the data if requested.
    def validation_batch_split(self, counter, dataset):
        x_train = []
        y_train = []
        x_validation = []
        y_validation = []
        count = counter % self.validation  # mod to handle the counter e.g. on 6 mod 3 is 2
        validation_data = dataset[count]
        count2 = 0
        for i in range(len(dataset)):
            if i != count:
                if count2 == 0:
                    train_data = dataset[i]
                else:
                    train_data = np.concatenate((train_data, dataset[i]))
                count2 += 1
        # # cant index array
        for row in train_data:
            x_train.append(row[:-1])
            y_train.append(row[-1])

        for row in validation_data:
            x_validation.append(row[:-1])
            y_validation.append(row[-1])

        if self.normalize.lower() == "true":
            x_train = np.asarray(x_train) / self.normalize_parameter
            x_validation = np.asarray(x_validation) / self.normalize_parameter

        y_train = np.array([[yii] for yii in y_train])
        y_validation = np.array([[yii] for yii in y_validation])

        batch_X = [x_train[i:i + self.batch_size] for i in range(0, len(x_train), self.batch_size)]
        batch_Y = [y_train[i:i + self.batch_size] for i in range(0, len(y_train), self.batch_size)]

        return x_validation, y_validation, batch_X, batch_Y, x_train, y_train, count

    # It takes as input the dataset in which it will apply the forward algorithm.
    # For every layer it computes the dot product + bias for every layer's weights, biases and activation functions
    # and it stores to weights_dict the value 'sum x' which is the product and the value 'forward x' as the activation
    # of that sum. Starts with the input and in every iteration the new data are the activation of the previous layer
    def forward_propagation(self, inputs):
        data = np.asarray(inputs)

        for index in range(len(self.hidden_layers_list) - 1):
            activation_function = self.input_parameters_dict['layer ' + str(index)]['activation']
            weights = self.weights_dict['layer ' + str(index)]['weights']
            bias = self.weights_dict['layer ' + str(index)]['bias']
            data_output = np.dot(data, weights) + bias.T
            self.weights_dict['sum ' + str(index)] = data_output

            data_output = activation_function(data_output)

            self.weights_dict['forward ' + str(index)] = data_output
            data = data_output

    # Takes as input the labels column and the current train set to compute the gradient of weights and biases.
    # Starts from the output layer and iterates through all the hidden layers to do the same procedure.
    def back_propagation(self, Y, x_t):
        m = len(x_t)  # Number of values used for averaging
        X = np.asarray(x_t)

        self.weights_dict['forward -1'] = X  # forward -1 is the x train

        activation_function_derivative = self.input_parameters_dict['layer ' + str(len(self.hidden_layers_list) - 2)][
            'activation derivative']

        # output layer gradient
        delta = np.multiply(-(2 / m) * (Y - self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)]),
                            activation_function_derivative(
                                self.weights_dict['sum ' + str(len(self.hidden_layers_list) - 2)]))
        grad = np.dot(self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 3)].T, delta)

        self.weights_dict['delta ' + str(len(self.hidden_layers_list) - 2)] = delta
        self.weights_dict['grad ' + str(len(self.hidden_layers_list) - 2)] = grad
        self.weights_dict['bias ' + str(len(self.hidden_layers_list) - 2)] = self.weights_dict[
            'delta ' + str(len(self.hidden_layers_list) - 2)].sum()

        # hidden layers gradient
        for index in range(len(self.hidden_layers_list) - 3, -1, -1):
            activation_function_derivative = self.input_parameters_dict['layer ' + str(index)]['activation derivative']

            delta = np.dot(delta,
                           self.weights_dict['layer ' + str(index + 1)]['weights'].T) * activation_function_derivative(
                self.weights_dict['sum ' + str(index)])
            self.weights_dict['delta ' + str(index)] = delta

            grad = np.dot(self.weights_dict['forward ' + str(index - 1)].T, delta)
            self.weights_dict['grad ' + str(index)] = grad

            self.weights_dict['bias ' + str(index)] = self.weights_dict['delta ' + str(index)].sum()

    # Initializes a list to calculate the momentum, a list for the rmsp, a list for the parameters (weights, biases)
    # a list for the gradients and a list to update the weights. Finally it updates the new weights and biases
    def Adam_optimizer(self):
        if self.time == 0:
            self.momentum = []
            self.rmsp = []
            self.params = []
            self.grad = []

            # append everything in the same order in order to process them simultaneously
            for index in range(len(self.hidden_layers_list) - 1):
                self.momentum.append(np.zeros_like(self.weights_dict['layer ' + str(index)]['weights']))
                self.momentum.append(np.zeros_like(self.weights_dict['layer ' + str(index)]['bias']))
                self.rmsp.append(np.zeros_like(self.weights_dict['layer ' + str(index)]['weights']))
                self.rmsp.append(np.zeros_like(self.weights_dict['layer ' + str(index)]['bias']))
                self.params.append(self.weights_dict['layer ' + str(index)]['weights'])
                self.params.append(self.weights_dict['layer ' + str(index)]['bias'])
                self.grad.append(self.weights_dict['grad ' + str(index)])
                self.grad.append(self.weights_dict['bias ' + str(index)])

        # every time it is called after the first time it updates the 2 lists with the new variables
        else:
            self.params = []
            self.grad = []
            for index in range(len(self.hidden_layers_list) - 1):
                self.params.append(self.weights_dict['layer ' + str(index)]['weights'])
                self.params.append(self.weights_dict['layer ' + str(index)]['bias'])
                self.grad.append(self.weights_dict['grad ' + str(index)])
                self.grad.append(self.weights_dict['bias ' + str(index)])

        self.time += 1
        for index in range(len(self.grad)):
            self.momentum[index] = self.beta1 * self.momentum[index] + (1 - self.beta1) * self.grad[index]
            self.rmsp[index] = self.beta2 * self.rmsp[index] + (1 - self.beta2) * (self.grad[index] ** 2)
            new_vars = (self.learning_rate * self.momentum[index]) / (np.sqrt(self.rmsp[index]) + self.epsilon)
            self.params[index] -= new_vars

        count = 0
        for i in range(len(self.hidden_layers_list) - 1):
            self.weights_dict['layer ' + str(i)]['weights'] = self.params[i + count]
            self.weights_dict['layer ' + str(i)]['bias'] = self.params[i + 1 + count]
            count += 1

    # User can choose on the loss function
    def loss_function(self, y, y_predicted):
        if self.loss_criterion == 'mse':
            return self.mse(y, y_predicted)
        elif self.loss_criterion == 'msle':
            return self.msle(y, y_predicted)
        else:
            sys.exit(
                "[ERROR] Loss function is not supported or is written wrong\nTry for input: mse or msle")

    # Mean squared error as loss function
    def mse(self, y, y_predicted):
        return ((y - y_predicted) ** 2).sum() / len(y)

    # Mean squared logarithmic error as loss function
    def msle(self, y, y_predicted):
        return np.mean((np.log(y + 1)) - np.log(y_predicted + 1))

    # sigmoid is the proposed output layer activation function because it outputs values on the [0,1] (we have
    # regression problem with 2 labels. ) . It rounds the output of sigmoid to get 1 or 0.
    # the output accuracy is the sum of the values predicted correct divided by the total number
    # The output is in % with 2 decimals accuracy
    def accuracy(self, y, y_predicted):
        y_predicted = np.round(y_predicted)  # 0.99 --> 1 prediction
        acc = round((np.sum(np.equal(y, y_predicted)) / len(y)) * 100, 2)

        return acc

    # Requires as input the epochs, validation loss, train loss, validation accuracy, train accuracy.
    # It plots in a 2x2 grid the last 4 in regards to the epochs
    def plot_metrics(self, ep, l_v, l_t, val, tra):
        plt.subplot(2, 2, 1)
        plt.plot(ep, l_t, 'k.-')
        plt.title('Train, Validation Loss-Epochs')
        plt.ylabel('Train Loss')

        plt.subplot(2, 2, 3)
        plt.plot(ep, l_v, 'g.-')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')

        plt.subplot(2, 2, 4)
        plt.plot(ep, val, 'r.-')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy %')

        plt.subplot(2, 2, 2)
        plt.plot(ep, tra, 'b.-')
        plt.title('Train, Validation Accuracy-Epochs')
        plt.ylabel('Train Accuracy %')

        # plt.savefig('loss-accuracy-plot.png', dpi=500)
        plt.show()

    # When called it calculates the forward for the data given, rounds them and saves them in a txt.
    def predict_test(self, test):
        self.forward_propagation(test)
        predicted_y = self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)]
        predicted_y = np.round(predicted_y)

        new_name = self.test_dir.replace('.csv', '_predicted.csv')

        np.savetxt(new_name, predicted_y, delimiter=',', fmt='%d')

        return predicted_y

    # This is the main function. It initializes some lists that will be used for plotting.
    # Initializes the self.hidden_layers_list, reads the data and initializes the weights.
    # For every epoch and for the batch_size given by the user it runs the process of training (forward propagation,
    # backwards for the gradient and finally Adam optimizer to update weights)
    # every compute_cost_acc_every_x_epochs it computes cost and accuracy
    # every change_validation_set_every_x_epochs it changes to the new validation fold
    def train(self):
        self.epoch_count = []
        self.loss_val_count = []
        self.loss_train_count = []
        self.accuracy_val_count = []
        self.accuracy_train_count = []
        self.prepare_input_parameters()

        dataset, x_test = self.read_data_and_map()

        self.initialize_weights()
        count = 0  # is the counter for the validation fold
        x_validation, y_validation, batch_X, batch_Y, x_train, y_train, number_of_fold = self.validation_batch_split(
            count, dataset)
        for epoch in range(self.epochs):
            for batch_index in range(len(batch_X)):
                self.forward_propagation(batch_X[batch_index])

                self.back_propagation(batch_Y[batch_index], batch_X[batch_index])

                self.Adam_optimizer()

            if epoch % self.metrics_change_time == 0:
                print(f'Epoch ====> {epoch}/{self.epochs}')
                self.forward_propagation(x_validation)
                val_acc = self.accuracy(y_validation,
                                        self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)])

                lf_v = self.loss_function(y_validation,
                                          self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)])

                self.forward_propagation(x_train)
                train_acc = self.accuracy(y_train,
                                          self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)])

                lf_t = self.loss_function(y_train,
                                          self.weights_dict['forward ' + str(len(self.hidden_layers_list) - 2)])

                print(f'Train accuracy is {train_acc} and Validation accuracy is {val_acc}')
                self.epoch_count.append(epoch)
                self.loss_val_count.append(lf_v)
                self.loss_train_count.append(lf_t)
                self.accuracy_val_count.append(val_acc)
                self.accuracy_train_count.append(train_acc)

            if epoch % self.validation_change_time == 0:
                count += 1  # update the counter of the validation fold
                x_validation, y_validation, batch_X, batch_Y, x_train, y_train, number_of_fold = self.validation_batch_split(
                    count, dataset)

        if self.test_data.lower() == "yes":
            self.predict_test(x_test)

        self.plot_metrics(self.epoch_count, self.loss_val_count, self.loss_train_count, self.accuracy_val_count,
                          self.accuracy_train_count)


class ActivationFunctions():
    '''
    A class that contains the activation functions and their derivatives
    '''

    def tanh(self, x):
        return np.tanh(x)

    def tanhDerivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(x, 0)

    def reluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-np.clip(x, -250, 250)))

    def sigmoidDerivative(self, x):
        return np.exp(-np.clip(x, -250, 250)) / (np.power(1. + np.exp(-np.clip(x, -250, 250)), 2))

    def leakyRelu(self, x):
        return np.maximum(0.5 * x, x)

    def leakyReluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    # it takes as input the parameters dictionary and maps each activation and their derivatives to the corresponding
    # layer
    def mapper(self, parameters: dict):
        for layer in parameters:
            if parameters[layer]['activation'] == 'relu':
                parameters[layer]['activation'] = ActivationFunctions().relu
                parameters[layer]['activation derivative'] = ActivationFunctions().reluDerivative

            elif parameters[layer]['activation'] == 'tangent':
                parameters[layer]['activation'] = ActivationFunctions().tanh
                parameters[layer]['activation derivative'] = ActivationFunctions().tanhDerivative

            elif parameters[layer]['activation'] == 'sigmoid':
                parameters[layer]['activation'] = ActivationFunctions().sigmoid
                parameters[layer]['activation derivative'] = ActivationFunctions().sigmoidDerivative

            elif parameters[layer]['activation'] == 'leakyRelu':
                parameters[layer]['activation'] = ActivationFunctions().leakyRelu
                parameters[layer]['activation derivative'] = ActivationFunctions().leakyReluDerivative

            else:
                sys.exit(
                    "[ERROR] Activation function is not supported or is written wrong\nTry for input: sigmoid, relu, tangent or leakyRelu")

        return parameters


if __name__ == "__main__":
    NN = NeuralNetworkRegressor(hidden_layers, hidden_layers_activation, output_layer_activation, epochs, train_dir,
                                test_dir, batch_size, learning_rate, beta1, beta2, epsilon, separator, headers,
                                validation, normalize, metrics_change_time, validation_change_time,
                                weights_init_Gaussian_mean, weights_init_Gaussian_standard_deviation, loss_criterion,
                                test_data)
    NN.train()
