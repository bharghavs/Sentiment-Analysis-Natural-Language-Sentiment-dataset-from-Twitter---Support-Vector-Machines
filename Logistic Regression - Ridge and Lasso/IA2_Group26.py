# AI534
# IA2 skeleton code

import math as ma
from ast import Num
import numpy as np
import pandas as pd
import matplotlib
from numpy import linalg as LA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


# By: Bharath, Cheng, Bharghav


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)
    return loaded_data


# Implements dataset preprocessing. For this assignment, you just need to implement normalization
# of the three numerical features.

def preprocess_data(data):
    # Your code here:
    data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['Annual_Premium'] = (data['Annual_Premium'] - data['Annual_Premium'].min()) / (
                data['Annual_Premium'].max() - data['Annual_Premium'].min())
    data['Vintage'] = (data['Vintage'] - data['Vintage'].min()) / (data['Vintage'].max() - data['Vintage'].min())
    preprocessed_data = data.to_numpy()

    return preprocessed_data


# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_data, val_data, lambda_val):
    # Your code here:
    lr = 1
    train_labels = train_data[:, -1]
    train_data = np.delete(train_data, -1, 1)
    val_labels = val_data[:, -1]
    val_data = np.delete(val_data, -1, 1)
    numFeat = np.shape(train_data)[1]
    numDataPoint = np.shape(train_data)[0]
    weights = np.random.rand(numFeat)
    count = 300
    losses = []
    # define two loss gradient to add a stopping a condition by comparing the difference of current and past loss gradient
    currLossGrad = np.random.rand(numFeat)
    lastLossGrad = np.random.rand(numFeat)

    while count > 0 and LA.norm(currLossGrad) > 1e-5 and LA.norm(currLossGrad) < 1e20:
        # helper function to calculate loss: this is not necessary by assignment but we want the loss to determine a good learning rate
        losses.append(L2_loss(train_data, train_labels, weights, numDataPoint, numFeat, lambda_val))
        # invoke a helper function to calculate loss gradient: both normal gradient descent + regularization are included in the helper function
        currLossGrad = L2_loss_grad(train_data, train_labels, weights, numDataPoint, numFeat, lambda_val)
        # if the gradient of loss does not change much, consider it converged
        if abs(LA.norm(currLossGrad) - LA.norm(lastLossGrad)) < 0.001:
            break
        lastLossGrad = currLossGrad
        weights -= currLossGrad * lr
        count -= 1
        # print(count)
    # plt_loss(losses)
    prediction_training = np.dot(train_data, weights)
    # invoke another helper function for calculating accuracy
    train_acc = accuracy(prediction_training, train_labels)
    prediction_val = np.dot(val_data, weights)
    val_acc = accuracy(prediction_val, val_labels)

    return weights, train_acc, val_acc


def logi(data, weight):
    return 1 / (1 + np.exp(-np.dot(data, weight)))


def accuracy(prediction, label):
    correct = 0
    for index in range(len(label)):
        if (label[index] == 1 and prediction[index] >= 0):
            correct += 1
        elif (label[index] == 0 and prediction[index] < 0):
            correct += 1
        else:
            correct += 0
    return correct / len(label)


def L2_loss(data, labels, weights, numDataPoint, numFeat, lambda_val):
    loss = 0
    # normal gradient descent
    for index in range(numDataPoint):
        loss -= (labels[index] * np.log(logi(data[index], weights)) + (1 - labels[index]) * np.log(
            1 - logi(data[index], weights)))
    loss /= numDataPoint
    # L2 regularization
    for feat in range(numFeat):
        loss += weights[feat] * weights[feat] * lambda_val
    return loss


def L2_loss_grad(data, labels, weights, numDataPoint, numFeat, lambda_val):
    grad = 0
    # normal gradient descent
    for index in range(numDataPoint):
        grad -= (labels[index] - logi(data[index], weights)) * data[index]
    grad /= numDataPoint
    # L2 regularization
    for feat in range(1, numFeat):
        grad[feat] += lambda_val * weights[feat]
    return grad


def L1_loss(data, labels, weights, numDataPoint, numFeat, lambda_val):
    loss = 0
    # normal gradient descent
    for index in range(numDataPoint):
        loss -= (labels[index] * np.log(logi(data[index], weights)) + (1 - labels[index]) * np.log(
            1 - logi(data[index], weights)))
    loss /= numDataPoint
    # L1 regularization
    for feat in range(numFeat):
        loss += abs(weights[feat]) * lambda_val
    return loss


def L1_loss_grad(data, labels, weights, numDataPoint):
    grad = 0
    # normal gradient descent
    for index in range(numDataPoint):
        grad -= (labels[index] - logi(data[index], weights)) * data[index]
    grad /= numDataPoint
    return grad


# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambda_val):
    # Your code here:
    lr = 0.01
    train_labels = train_data[:, -1]
    train_data = np.delete(train_data, -1, 1)
    val_labels = val_data[:, -1]
    val_data = np.delete(val_data, -1, 1)
    numFeat = np.shape(train_data)[1]
    numDataPoint = np.shape(train_data)[0]
    weights = np.random.rand(numFeat)
    count = 800
    losses = []
    currLossGrad = np.random.rand(numFeat)
    lastLossGrad = np.random.rand(numFeat)

    while count > 0 and LA.norm(currLossGrad) > 1e-5 and LA.norm(currLossGrad) < 1e20:
        # helper function to calculate loss: this is not necessary by assignment but we want the loss to determine a good learning rate
        losses.append(L1_loss(train_data, train_labels, weights, numDataPoint, numFeat, lambda_val))
        # calculating the gradient of loss: L1 is not differentiable, ignore it
        currLossGrad = L1_loss_grad(train_data, train_labels, weights, numDataPoint)
        lastLossGrad = currLossGrad
        # normal gradient descent
        weights -= currLossGrad * lr
        # proximal gradient descent for L1 regularization
        for feat in range(1, numFeat):
            weights[feat] = np.sign(weights[feat]) * max(0, abs(weights[feat]) - lr * lambda_val)
        count -= 1
        # print(count)
    # plt_loss(losses)
    prediction_training = np.dot(train_data, weights)
    train_acc = accuracy(prediction_training, train_labels)
    prediction_val = np.dot(val_data, weights)
    val_acc = accuracy(prediction_val, val_labels)
    return weights, train_acc, val_acc


# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda_val values and then put multiple loss curves in a single plot.
def plot_losses(accs):
    # Your code here:

    return


# plot loss to determine a learning rate
def plt_loss(losses):
    xAxis = np.arange(0, len(losses))
    # asked by the assignment to plot y axis in log scale
    yAxis = losses
    plt.plot(xAxis, yAxis)
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.savefig('lr1e_1lambda1e_1.png')
    plt.close()
    # plt.show()
    return losses


# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:

lambda_array = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambda_vals
# Your code here:
L2_train_output = []
L2_val_output = []
L2_trained_weights_abs = []
L2_sparsities = []

for lambda_val in lambda_array:
    loaded_train_data = load_data("IA2-train.csv")
    pre_proc_train_data = preprocess_data(loaded_train_data)
    loaded_val_data = load_data("IA2-dev.csv")
    pre_proc_val_data = preprocess_data(loaded_val_data)
    num_non_zeros = 0

    L2_trained_weights, L2_trained_acc, L2_valed_acc = LR_L2_train(pre_proc_train_data, pre_proc_val_data, lambda_val)
    L2_train_output.append(L2_trained_acc)
    L2_val_output.append(L2_valed_acc)
    
    # Sparsity
    for weight in L2_trained_weights:
        if weight > ma.pow(10,-6):
            num_non_zeros += 1

    sparsity = 1 - (num_non_zeros / float(L2_trained_weights.size))
    L2_sparsities.append(sparsity)
    print("L2: Sparsity for lambda " + str(lambda_val) + ": " + str(sparsity))
    
    # Top 5 features with max weight
    L2_trained_weights_abs = np.absolute(L2_trained_weights)
    print("L2: Lambda value is: ", lambda_val)
    ind = np.argpartition(L2_trained_weights_abs[1:], -5)[-5:]
    ind = ind + 1
    print("top 5 index are: ", ind)
    top5 = L2_trained_weights_abs[ind]
    print("wt of top 5 are:", top5)

print("train and vali accuracy for L2: ", L2_train_output, L2_val_output)

x = np.arange(len(lambda_array))
width = 0.35

fig, ax = plt.subplots()
L2_train = ax.bar(x - width / 2, L2_train_output, width, label='Training Accuracy')
L2_val = ax.bar(x + width / 2, L2_val_output, width, label='Validation Accuracy')

ax.set_xticks(x)
ax.set_xticklabels(lambda_array)
ax.set_title('LR with L2 (Ridge) regularization')
ax.set_ylabel("Accuracy")
ax.set_xlabel("Lambda")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('L2:Accuracy Plot.png')

fig, ax = plt.subplots()
L2_train = ax.bar(x - width / 2, L2_sparsities, width, label='Sparsities')

ax.set_xticks(x)
ax.set_xticklabels(lambda_array)
ax.set_title('L2:Sparsities')
ax.set_ylabel("Sparsity")
ax.set_xlabel("Lambda")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('L2:Sparsities.png')

# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:
L2_noisy_train_output = []
L2_noisy_val_output = []

for lambda_val in lambda_array:
    loaded_train_data = load_data("IA2-train-noisy.csv")
    pre_proc_train_data = preprocess_data(loaded_train_data)
    loaded_val_data = load_data("IA2-dev.csv")
    pre_proc_val_data = preprocess_data(loaded_val_data)
    
    L2_noisy_trained_weights, L2_noisy_trained_acc, L2_noisy_valed_acc = LR_L2_train(pre_proc_train_data, pre_proc_val_data, lambda_val)
    L2_noisy_train_output.append(L2_noisy_trained_acc)
    L2_noisy_val_output.append(L2_noisy_valed_acc)
    
print("train and vali accuracy for L2-noisy: ", L2_noisy_train_output, L2_noisy_val_output)


x = np.arange(len(lambda_array))
width = 0.35

fig, ax = plt.subplots()
L2_noisy_train = ax.bar(x - width/2, L2_noisy_train_output, width, label='Training Accuracy')
L2_noisy_val = ax.bar(x + width/2, L2_noisy_val_output, width, label='Validation Accuracy')

ax.set_xticks(x)
ax.set_xticklabels(lambda_array)
ax.set_title('LR with L2 (Ridge) regularization with Noisy data')
ax.set_ylabel("Accuracy")
ax.set_xlabel("Lambda")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('L2:Accuracy Plot-Noisy Data.png')

# Part 3  Implement logistic regression with L1 regularization and experiment with different lambda_vals
# Your code here:
L1_train_output = []
L1_val_output = []
L1_trained_weights_abs = []
L1_sparsities = []

for lambda_val in lambda_array:
    loaded_train_data = load_data("IA2-train.csv")
    pre_proc_train_data = preprocess_data(loaded_train_data)
    loaded_val_data = load_data("IA2-dev.csv")
    pre_proc_val_data = preprocess_data(loaded_val_data)
    num_non_zeros = 0
    
    L1_trained_weights, L1_trained_acc, L1_valed_acc = LR_L1_train(pre_proc_train_data, pre_proc_val_data, lambda_val)
    L1_train_output.append(L1_trained_acc)
    L1_val_output.append(L1_valed_acc)
    
    # Sparsity
    for weight in L1_trained_weights:
        if weight > ma.pow(10,-6):
            num_non_zeros += 1

    sparsity = 1 - (num_non_zeros / float(L1_trained_weights.size))
    L1_sparsities.append(sparsity)
    print("L1: Sparsity for lambda " + str(lambda_val) + ": " + str(sparsity))
    
    # Top 5 weights
    L1_trained_weights_abs = np.absolute(L1_trained_weights)
    print("L1: Lambda value is: ", lambda_val)
    ind = np.argpartition(L1_trained_weights_abs[1:], -5)[-5:]
    ind = ind + 1
    print("top 5 index are: ", ind)
    top5 = L1_trained_weights_abs[ind]
    print("wt of top 5 are:", top5)
    
print("train and vali accuracy for L1: ", L1_train_output, L1_val_output)


x = np.arange(len(lambda_array))
width = 0.35

fig, ax = plt.subplots()
L1_train = ax.bar(x - width/2, L1_train_output, width, label='Training Accuracy')
L1_val = ax.bar(x + width/2, L1_val_output, width, label='Validation Accuracy')

ax.set_xticks(x)
ax.set_xticklabels(lambda_array)
ax.set_title('LR with L1 (Lasso) regularization')
ax.set_ylabel("Accuracy")
ax.set_xlabel("Lambda")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('L1:Accuracy Plot.png')

fig, ax = plt.subplots()
L1_train = ax.bar(x - width / 2, L1_sparsities, width, label='Sparsities')

ax.set_xticks(x)
ax.set_xticklabels(lambda_array)
ax.set_title('L1:Sparsities')
ax.set_ylabel("Sparsity")
ax.set_xlabel("Lambda")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('L1:Sparsities.png')
