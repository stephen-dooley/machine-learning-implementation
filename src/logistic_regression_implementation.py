
# coding: utf-8

# #### User Input and Utility Functions

# In[125]:

import sys
import numpy
import numpy as np
import csv
from decimal import Decimal
import random as rnd
import pandas as pd
numpy.seterr(all='ignore')

# import csv to visualise data
df = pd.read_csv('../data/owls-csv.csv')

# get the user input
number_of_classes = int(input('How many types of owls are there?\n'));
##################################### UTILITY FUNCTIONS ######################################

# file_reader 
#
# Reads in the data file from local storage
#
# @params {String} file_name - location of file on disk
def file_reader(file_name):
    inputs = [];
    i = 0;
    with open(file_name, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter='\n')
        for row in file:
            split_row = row[0].split(',')
            floats = [float(x) for x in split_row]
            inputs.append(floats)
    # return the input array
    return inputs;


# normalise_data
#
# Normalises the data inputs for the model to erradicate the effect of outliers.
# The data is normalised with 0-1 normalisation by dividing the data for each 
# instance by the max value for that instance.
#
# @params {List} data - list of lists, containing the data about each instance
def normalise_data(data):
    # number of intances in dataset
    numberOfInstances = len(data)
    # get min/max values from the dataset
    max_values = map(max, zip(*data));
    
    max_values_array = [];
    for val in max_values:
        max_values_array.append(val);
    max_values_array = [float(i) for i in max_values_array]
    
    for i in range(0, numberOfInstances):
        #normalize each element of the lists
        x = 0
        while x < len(data[i]):
            # 0-1 normalisation
            # divide each element by the max value for that attribute in set
            data[i][x] = data[i][x] / max_values_array[x];
            x = x + 1;
    
    return np.asarray(data);
 
    
# softmax
#
# Logisitc Regression function used to restrict the outputs to a 
# range of 0 --> 1. The outputs can then be interpretted as a 
# probability or odds.
#
# @params {list} x - data to train/test on 
def softmax(x):
    # e = numpy.exp(x - numpy.max(x))  # prevent overflow
    e = numpy.exp(x)
    # when there is only 1 instance to test
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    # testing multiple instances
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # number of dimensions = 4

# Percentage Split Data
#
# Split the data by a given percentage based on whether you are training 
# or testing the model
#
# object - contains the parameters that define the Percentage Split Data Class
def percentage_split_data(data, locations, percentage):
    sample_locations = locations;
    
    if percentage: 
        # get the number of samples needed to make up specified percentage
        number_of_samples = int( len(data)*(percentage/100) )
        # list of indices to sample the data with
        sample_locations = rnd.sample(range(len(data)), number_of_samples)
        
    split_list = []
    for index, row in enumerate(data):
        # if the index is found in the list of random indices, take the instance
        if index in sample_locations:
            split_list.append(row)

    return split_list;

# convert_predictions_binary
#
# Convert the highest predicted probabilities to 1
# otherwise set as 0
#
# @params {list} results - results from testing model
def convert_predictions_binary(results):
    converted_rersult = []
    for row in results:
        # create array of zeros for the number of classes
        zeros = [0] * number_of_classes;
        # find the max probability of the set
        for index, el in enumerate(row):
            if el == max(row):
                index_of_max = index;
        zeros[index_of_max] = 1;
        
        converted_rersult.append(zeros);
                
    return converted_rersult;

# create_output_string
#
# Create maniplative string to write to file
# Makes it easier to read back the results
#
# @params {list} data - results from testing model
def create_output_string(data):
    # create string to save to output
    output_string = ''
    for index1, test in enumerate(data):
        string = ''
        for index2, val in enumerate(test):
            int(val)
            string += str(int(val))
            if index2 < len(data[0])-1:
                string += ','

        output_string += string
        output_string += '|'
    
    return output_string;

#############################################################################################


# #### Logistic Regression Class and Test Functionality

# In[151]:


##################################### REGRESSION CLASS ######################################

# Logistic Regression
#
# Inner class to build the Logistic Regression model.
# It trains the data on 66% and tests on 33%.
#
# object - contains the parameters that define the Logistic Regression Class
class LogisticRegression(object):
    def __init__(self, input, type_of_owl, n_in, n_out):
        self.x = input                       # input values of owls
        self.y = type_of_owl                 # output values 
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0
        self.params = [self.W, self.b]

    # train the model
    # lr = learning rate
    def train_model(self, lr=0.015, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # find the predicted probabilties by running the sum of vector product of the input data
        # & array of zeros and an array of zeros.
        # on each iteration the value of self.W and self.b are calculated below
        prob_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        # calculate the difference in the predited probability and the actual value
        diff_y = self.y - prob_y_given_x
        # append to the value of self.W and self.B on every iteration
        self.W += (lr * numpy.dot(self.x.T, diff_y)) - (lr * L2_reg * self.W)
        # multiply the learning rate by the mean of the difference in the actual vs. prediction
        self.b += lr * numpy.mean(diff_y, axis=0)
    
    
    # test the model
    def predict_probabilities(self, x):
        return softmax(numpy.dot(x, self.W) + self.b)
#############################################################################################



####################################### TEST FUNCTION ########################################
def test_algorithm(learning_rate=0.015, n_interations=1000):
    training_percentage = 66
    testing_percentage = 100 - training_percentage
    
    # training data
    input_data = numpy.array( file_reader('../data/input-data') );
    # get the number of samples needed to make up specified percentage
    number_of_samples = int( len(input_data)*(training_percentage/100) )
    # list of indices to sample the data with
    sample_locations = rnd.sample(range(len(input_data)), number_of_samples)
    
    # get percentage of dataset for training
    sampled_input_data = percentage_split_data(input_data, sample_locations, None)
    # normalize the data
    x = normalise_data(sampled_input_data)
    
    # types of owl
    owl_types = numpy.array( file_reader('../data/owl-types') );
    # outputs: (data, percentage split)
    y = percentage_split_data(owl_types, sample_locations, None)

    # build LogisticRegression model
    LogisticRegressionModel = LogisticRegression(input=x, type_of_owl=y, n_in=(len(df.columns)-1), n_out=number_of_classes)

    #### TRAIN MODEL ####
    for iteration in range(n_interations):
        LogisticRegressionModel.train_model(lr=learning_rate)
        learning_rate *= 0.96
                

    #### TEST MODEL ####
    # get the locations of the samples not used in the training of the model
    # and use them for testing the model. 
    test_sample_locations = []
    for location in range(135):
        if location not in sample_locations:
            test_sample_locations.append(location);

    # sample the data for testing
    testing_data_x = percentage_split_data(input_data, test_sample_locations, None)
    # normalize the data
    normalised_testing_data_x = normalise_data(testing_data_x)
    # run the prediction model on the data
    prediction_model = LogisticRegressionModel.predict_probabilities(normalised_testing_data_x)
    # convert the highst probabiility to a 1 and the others to a 0
    probabilities = convert_predictions_binary(prediction_model)

    # compare the results to the actual value of the 
    testing_data_y = percentage_split_data(owl_types, test_sample_locations, None)
    
    
    output_string = create_output_string(probabilities)
    output_string = output_string[:-1]
    with open('../data/results/predictions', 'a') as predictions:
        predictions.write(output_string)
        predictions.write('\n')
        
    output_string = create_output_string(testing_data_y)
    output_string = output_string[:-1]
    with open('../data/results/actual-probabilities', 'a') as predictions:
        predictions.write(output_string)
        predictions.write('\n')
    
#############################################################################################


# #### Run Model

# In[152]:

####################################### MAIN PROGRAM ########################################
if __name__ == "__main__":
    test_algorithm()
    print('Model Built\nPredictions Made');
#############################################################################################

