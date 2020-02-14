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
learning_rate = 0.1

# create instance of neural network
n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# epoches is the number of times the training data set is used for training
epoches = 5

# train the neural network
for e in range(epoches):
# go through all records in the training data set
    for record in training_data_list:
        # split the record by ','
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass
pass

# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test_10.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural work
# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
    # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
    # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
pass

# calculate the performance score, the fraction of correct answers

scorecard_array = numpy.asarray(scorecard)
print (scorecard_array)
print ("performance = ", scorecard_array.sum() /
scorecard_array.size)
