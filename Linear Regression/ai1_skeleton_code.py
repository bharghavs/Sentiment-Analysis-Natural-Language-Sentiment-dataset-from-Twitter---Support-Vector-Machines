# AI 534 - Solo Assignment
# AI1 skeleton code
# By Bharghav Srikhakollu

from ast import Num
import numpy as np
import pandas as pd
import matplotlib
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv



# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    #loaded_data = np.genfromtxt(path, delimiter=',')
    df_from_csv = pd.read_csv(path)
    return df_from_csv

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15, datatype):
    # Your code here:
    # Split date to YYMMDD
    data[["month", "day", "year"]] = data["date"].str.split("/", expand=True)
    data['year'] = data['year'].astype(int)
    data['month'] = data['month'].astype(int)
    data['day'] = data['day'].astype(int)
    #print(pd.DataFrame(data))
    preprocessed_data = pd.DataFrame(data).to_numpy()
    # Delete the column of house ID, and the column of date
    preprocessed_data = np.delete(preprocessed_data, [0,1], 1)
    
    # Add a constant "1" column
    numRow = np.shape(preprocessed_data)[0]
    preprocessed_data = np.c_[preprocessed_data, np.ones(numRow)]
    #  yr renovated to "age since built"
    for index in range(numRow):
        if preprocessed_data[index][12] == 0:
            preprocessed_data[index][12] = preprocessed_data[index][-2] - preprocessed_data[index][11]
        else:
            preprocessed_data[index][12] = preprocessed_data[index][-2] - preprocessed_data[index][12] 
    
    for index in range(numRow):
        preprocessed_data[index][7] = preprocessed_data[index][7] / 5
        preprocessed_data[index][8] = preprocessed_data[index][8] / 13
# remove redundant livingsqft if being asked
#    if drop_sqrt_living15 == 1:
    preprocessed_data = np.delete(preprocessed_data, [16,17], 1)
    # Calculate mean value by col
    meanVal = np.mean(preprocessed_data, dtype=np.float64, axis=0)
    # Calculate std value by col
    stdVal = np.std(preprocessed_data, dtype=np.float64, axis=0)
    if normalize == 1:
#        if drop_sqrt_living15 == 0:
#            # Normalize each column by substracting to mean before diving by std
#            for col in range(23):
#                # Do not implement norm on waterfront, price, and constant 1
#                if col != 5 and col != 18 and col!= 22:
#                    preprocessed_data[:,col] -= meanVal[col]
#                    preprocessed_data[:,col] /= stdVal[col]
#        else:
        if datatype == "train":
            for col in range(21):
                # Do not implement norm on waterfront, price, and constant 1
                if col != 5 and col != 16 and col!= 20:
                    preprocessed_data[:,col] -= meanVal[col]
                    preprocessed_data[:,col] /= stdVal[col]  
        else:
            for col in range(20):
                # Do not implement norm on waterfront, and constant 1
                if col != 5 and col!= 19:
                    preprocessed_data[:,col] -= meanVal[col]
                    preprocessed_data[:,col] /= stdVal[col]       

    return preprocessed_data




# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:
    # separate labels from data
    labels = data[:, 16]
    data = np.delete(data, 16, 1)
    # Number of features = number of columns from pre-processed data 
    numFeat = np.shape(data)[1] 
    # Number of training data = number of rows from pre-processed data \
    numDataPoint = np.shape(data)[0] 
    # Initialize weights to all zero 
    weights = np.random.rand(numFeat)
    # this array stores the loss after each iteration
    losses = []
    #we are supposed to run 4000 iterations if the loss is not convereged 
    count = 4000
    # this record loss gradient for the current iteration
    currLossGrad = np.random.rand(numFeat)
    lastLossGrad = np.random.rand(numFeat)
    # termination conditions
    while count > 0 and LA.norm(currLossGrad) > 1e-5 and LA.norm(currLossGrad) < 1e20:
        # TODO: calculate current loss and gradient
        losses.append(loss(data, labels, weights, numDataPoint))
        currLossGrad = lossGrad(data, labels, weights, numDataPoint, numFeat)
        if abs(LA.norm(currLossGrad) - LA.norm(lastLossGrad)) < 0.001:
            break
        lastLossGrad = currLossGrad
        weights -= currLossGrad * lr
        count -= 1
        print(count)
    return weights, losses

def loss(data, labels, weights, numDataPoint):
    MSE = 0
    for index in range(numDataPoint):
        MSE += np.square(np.dot(data[index], np.transpose(weights)) - labels[index])
    return MSE/numDataPoint

def lossGrad(data, labels, weights, numDataPoint, numFeat):
    
    grad = data[0] * (np.dot(data[0], np.transpose(weights)) - labels[0])

    for index in range(1, numDataPoint):
        #print(np.dot(data[index], np.transpose(weights)) - labels[index])
        grad +=  data[index] * (np.dot(data[index], np.transpose(weights)) - labels[index])
    grad *= 2/numDataPoint
    grad = grad.astype(np.float64)
    #print("res",grad)   
    return grad    


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:
    xAxis = np.arange(0, len(losses))
    #asked by the assignment to plot y axis in log scale 
    yAxis = losses
    plt.plot(xAxis, yAxis)
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.savefig('RedundancyNormalized_1e_2.png')
    plt.close() 
    #plt.show()   
    return losses


# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
preprocessed_data = preprocess_data(load_data('PA1_train1.csv'), 1, 1, "train")

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:
weights, losses = gd_train(preprocessed_data, 1, 0.01)
plot_losses(losses)
testing_preprocessed_data = preprocess_data(load_data('PA1_test1.csv'), 1, 1, "test")
# separate labels from data
#testing_labels = testing_preprocessed_data[:, 18]
#testing_preprocessed_data = np.delete(testing_preprocessed_data, 18, 1)
# Number of features = number of columns from pre-processed data 
testing_numFeat = np.shape(testing_preprocessed_data)[1] 
# Number of training data = number of rows from pre-processed data \
testing_numDataPoint = np.shape(testing_preprocessed_data)[0]
#testing_MSE = loss(testing_preprocessed_data, testing_labels, weights, testing_numDataPoint)
predict = np.dot(testing_preprocessed_data, weights)
print(predict)
#print(testing_MSE)
#print(weights)
#with open('output.txt', 'w') as f:
#    for num in predict:
#        f.write(str(num))     
a = np.asarray(predict)
np.savetxt("foo.txt", a, delimiter = " ")
# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here: