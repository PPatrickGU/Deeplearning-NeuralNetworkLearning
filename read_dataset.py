#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
import matplotlib.pyplot as plt
import neural_network as nn

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.3

# create instance of neural network
n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# go through all records in the training data set
for record in training_data_list:
    # split the record by ','
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #print(len(inputs))
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    #print(len(targets))
    n.train(inputs, targets)
    pass

#plt.imshow(image_array, cmap='Greys', interpolation='None')
#plt.show()
#print(len(data_list))
#print(data_list[0])


#print(scaled_input)

#output nodes is 10 (example)
