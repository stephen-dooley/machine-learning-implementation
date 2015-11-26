
# coding: utf-8

# <h2 align="center">Implementation of the Logistic Regression Classification Algorithm</h2>
# <h4 align="center">Machine Learning & Data Mining - Assignment III</h4>
# <h4 align="center">Stephen Dooley - 12502947 - 26/11/15</h4> 

# ### (1) Import Dataset

# In[23]:

import pandas as pd
# import csv to visualise data
df = pd.read_csv('../data/owls-csv.csv')

# get the user input
number_of_classes = int(input('How many types of owls are there?\n'));


# ### (2) Graphing the Dataset
# 
# The data is divided into each of the following types of owl:
# * Long Eared Owl
# * Snowy Owl
# * Barn Owl
# 
# Each instance contains data about the body length, wing length, body width and wing width of a given owl. The plot below illustrates the categorical data by dividing each of the owl types by a line separator. See the legend on the plot for more information.

# In[24]:

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().magic('pylab inline')

# get user input 
y_max = int(input('Enter preferred max value for y-axis:\n(For current dataset enter 9)\n'));

# generify plot variable to work for different number of attributes
number_of_instances = len(df.type);
instances_per_class = number_of_instances/number_of_classes;

# initliase the dividers for the graph
# helps to build a visualisation of the classes
x_divider_one = [instances_per_class]*number_of_instances; 
x_divider_two = [instances_per_class*2]*number_of_instances;
y_divider = np.arange(0, y_max, y_max/number_of_instances);

# setup figure
fig = plt.figure(figsize=(20, 6));
ax1 = fig.add_subplot(111);
# plot the data 
cm_bright = ListedColormap(['#FF0000', '#0000FF']);

''' scatter(x, y, marker size, color, marker color style, label) '''
ax1.scatter(np.arange(number_of_instances), df.body_length, 
            color='red', cmap=cm_bright, label='body_length');
ax1.scatter(np.arange(number_of_instances), df.wing_length, 
            color='black', cmap=cm_bright, label='wing_length'); 
ax1.scatter(np.arange(number_of_instances), df.body_width, 
            color='blue', cmap=cm_bright, label='body_width'); 
ax1.scatter(np.arange(number_of_instances), df.wing_width, 
            color='green', cmap=cm_bright, label='wing_width');
ax1.scatter(x_divider_one, y_divider, s=1, color='black', cmap=cm_bright);
ax1.scatter(x_divider_two, y_divider, s=1, color='black', cmap=cm_bright);
ax1.set_xlim(0, number_of_instances)
ax1.set_ylim(0, y_max);
ax1.set_xlabel('Instance');
ax1.set_ylabel('Length or Width');
ax1.legend(loc='upper left');


# ### (3) Data Observations and Test Plan
# As seen from the plot above, the attributes of the Long Eared Owl (left), are easily differentiated from the those of the Snowy Owl (middle) and Barn Owl (right). It is observed that the body length for all three types of owl is similar. Although all three types of owl share similar body length it appears as though error in the algorithm may arise when trying to distinguish between the Snowy Owl and Barn Owl. These two types of owl contain similar ranges of values for body length, wing length, body width and wing width. Training the model to distinguish the difference between the Snowy Owl (middle) and Barn Owl (right) will be the most difficult due to the similarities in their attributes.
# 
# In logistic regression, the predictions of the model are constrained to the range [0,1]. This is to allow the value to be interpreted as an estimation of the probability. The estimated probability is used to predict which type of owl the instance in question belongs to. By using the logit function (inverse of logistic function), it is possible to translate the predictions from the range [−∞,∞] to [0,1]. It translates a *K*-dimensional vector $x$ of real values to a *K*-dimensional vector $\sigma(\mathbf{x})$ of real values in the range (0, 1). $\sigma(\mathbf{x})$ represents the predicted probabilities, and the sum of all probabilities of the vector $\sigma(\mathbf{x})$ add up to 1. The logistic function (or softmax function for multiclass tasks) of a number x between 0 and 1 is given by:
# 
# $$ \sigma(\mathbf{x})_j = \frac{e^{\mathbf{x}k}}{\sum_{k}^{K} e^{\mathbf{x}k}} $$
# 
# The graph of the logit function can be seen below:

# In[25]:

x = np.linspace(-10, 10, 100)
y = 1.0 / (1.0 + np.exp(-x))
ax2 = plt.subplot(111)
ax2.plot(x, y, 'r-', label='logit')
ax2.set_ylabel('Probability');
ax2.legend(loc='lower right')


# ### (4) Softmax vs. Sigmoid
# The sigmoid function is used for binary classification. The sigmoid function is given by: $ \sigma(x)_j = \frac{1}{1 + e^{-x}} $
# 
# The softmax function is used for multi-class problems such as this one (3 types of owl). The softmax function is given by (for vector x): $ \sigma(\mathbf{x})_j = \frac{e^{\mathbf{x}k}}{\sum_{k}^{K} e^{\mathbf{x}k}} $
# 
# Both functions bind the predicted probabilities to the range [0,1]. The softmax function is an adaptation of the sigmoid, whereby the vector sum of the exponentials is the denominator.
# 
# Two factors that heavily influence the performance of the model are the learning rate and the number of epochs the model is run for whilst training. The lower the learning rate, the slower it learns. With low learning rates it is easier to observe the changes in the predicted probabilities as the model is trained. The number of iterations will affect the speed and accuracy of the model also. Running iterations infinitely will be worthless as the accuracy will plateau eventually. For this model, the accuracy plateaued at approximately 500 iterations.

# ### (5) Prepare Data
# The data is divided into 2 files:
# * The raw input data for each instance eg. [body length, body width, wing length, wing width]
# * The type of owl for each instance.
# 
# The input data is normalised by dividing each attribute (eg. body length) by the max value for the same attribute for all instances. This can be done for datasets with all positive values like the owl dataset. Normalising the data helps to produce a prediction that is easily interpreted, while reducing the effect of outliers in the dataset.
# 
# The type of owl was transformed into a binary like array. For example, the if the owl was in the first group (Long Eared Owl), the data for that instance is $[1, 0, 0]$. The values in the array represent the probability of being a [Long Eared Owl, Snowy Owl, Barn Owl] in that order. The sum of the array should always equal 1 (max probability). Similarly, the predicted probabilities would take the form of an array. The sum of the predicted probabilities is also equal to 1.
# 
# The dataset contains 135 instances which is split 2/3 for training the model, and 1/3 for testing the model. The data for testing is entirely exclusive from the training data. All data used for training is omitted when testing.

# ### (6) Testing and Results
# The precision of the algorithm is given by dividing the True Positives (TP) plus the True Negatives (TN) divided by the total number of instances tested. 20 tests were carried out. Each test was run for 500 iterations. On each iteration the difference between the predicted probability and actual probability is minimized. 

# In[26]:

# read the results in from the file
with open('../data/results/predictions') as predictions:
    for line in predictions:
        list_of_predictions = [line.split('|') for row in predictions]

with open('../data/results/actual-probabilities') as actual:
    for line in actual:
        list_of_actual = [line.split('|') for row in actual]

print('Calculating the number of true positives for each of the %s tests...' % len(list_of_actual))
print('Calculating the average accuracy for tests...')

num_of_instances = 0
precision = []
matches = 0
# get the number of TP per test and and calculate average accuracy
for index, test in enumerate(list_of_predictions):
    num_of_instances = len(test)
    matches = 0
    for x, string in enumerate(test):
        if(list_of_predictions[index][x] == list_of_actual[index][x]):
            matches += 1;
            
    precision.append(matches);
    
precision_of_algoirthm = (sum(precision)/len(precision))/num_of_instances*100
print('Accuracy of the algorithm on testing is %s' % precision_of_algoirthm, '%')


# ### (7) Conclusions and Observations
# 
# As expected, the model performed very well when predicting the probability of an owl belonging to the first group (see section (3) Data Observations and Test Plan). When training the algorithm, there were only 89 instances (66% of the dataset). In general, it consisted of approximately 30 instances of each owl. This means there were only 30 cases of the Snowy Owl and 30 cases of the Barn Owl for the model to train with. If the dataset was larger, it may have been more accurate at detecting the difference between these 2 types. Due to their similarities and lack of training instances, the algorithm struggled when trying to predict the probability for the second and third type. 

# ### (8) How to Run the Program
# 
# * Download source code and run *logistic_regression_implementation.py*
# * Follow on-screen prompts for user inputs
# * Once the model has been trained and tested, preview the accuracy by running the program stephen_dooley_report.py

# This model was built on iPython notebooks by Jupyter.

# ### Appendix

# In[ ]:

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
    LogisticRegressionModel = LogisticRegression(input=x, type_of_owl=y, 
                                                 n_in=(len(df.columns)-1), 
                                                 n_out=number_of_classes)

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


####################################### MAIN PROGRAM ########################################
if __name__ == "__main__":
    test_algorithm()
    print('Model Built\nPredictions Made');
#############################################################################################

