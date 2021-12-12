import pandas as pd
import numpy as np
import csv
import sympy as syn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('hi')

# IMPORT THE DATA
train_labels = []
count = 0
with open('question-3-labels-train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            train_labels.append(row)
train_labels = np.array(train_labels)
train_labels = train_labels.astype(np.float64)

train_features = []
count = 0
with open('question-3-features-train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            train_features.append(row)
train_features = np.array(train_features)
train_features = train_features.astype(np.float64)


test_features = []
count = 0
with open('question-3-features-test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            test_features.append(row)
test_features = np.array(test_features)
test_features = test_features.astype(np.float64)


test_labels = []
count = 0
with open('question-3-labels-test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            test_labels.append(row)
test_labels = np.array(test_labels)
test_labels = test_labels.astype(np.float64)



# NORMALIZATION
train_features[:, 0] = (train_features[:, 0] - min(train_features[:, 0])) / (
            max(train_features[:, 0]) - min(train_features[:, 0]))
train_features[:, 1] = (train_features[:, 1] - min(train_features[:, 1])) / (
            max(train_features[:, 1]) - min(train_features[:, 1]))
train_features[:, 2] = (train_features[:, 2] - min(train_features[:, 2])) / (
            max(train_features[:, 2]) - min(train_features[:, 2]))

lr = [1 / 100000, 1 / 10000, 1 / 1000, 1 / 100, 1 / 10]
bias = 0

# FULL BATCH GRADIENT DECENT
def sigmoid(eq):
    return 1 / (1 + np.exp(-eq))

weights = np.zeros(3)
for i in range(1000):
    y_pred = sigmoid((train_features.dot(weights)) + bias)
    dw = (1 / len(train_labels)) * np.dot(train_features.T, (y_pred - train_labels))
    db = (1 / len(train_labels)) * np.sum(y_pred - train_labels)
    new_dw = [dw[0][0], dw[1][0], dw[2][0]]
    count = 0
    while count < 3:
        weights[count] = weights[count] - new_dw[count] * lr[1]
        count = count + 1
    bias = bias - lr[1] * db
print(weights)

predictions = []
for i in range(len(test_features)):
    y_pred = 0
    z = weights[0] * test_features[i][0] + weights[1] * test_features[i][1] + weights[2] * test_features[i][2] + bias
    y_pred = sigmoid(z)
    #print(y_pred)
    if y_pred > 0.4036:
        Y = 1
    else:
        Y = 0
    predictions.append(Y)

# PERFORMANCE METRICS OF FULL BACTH
accuracy = 0
count = 0
TN = 0
TP = 0
FP = 0
FN = 0

for i in range(len(test_labels)):
    if test_labels[i] == int(predictions[i]):
        count = count + 1
    if test_labels[i] == 0 and predictions[i] == 0:
        TN = TN + 1
    if test_labels[i] == 1 and predictions[i] == 1:
        TP = TP + 1
    if test_labels[i] == 1 and predictions[i] == 0:
        FN = FN + 1
    if test_labels[i] == 0 and predictions[i] == 1:
        FP = FP + 1

accuracy = count / len(test_labels) * 100
print("Calculate the metrics of full-batch ")
print("Accuracy = " + str(accuracy))
print("TP : ", TP, "FP: ", FP, "TN :", TN, "FN :", FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
NPV = TN / (TN + FN)
FPR = FP / (FP + TN)
FDR = FP / (TP + FP)
F1 = (2 * precision * recall) / (precision + recall)
F2 = (5 * precision * recall) / (4 * precision + recall)

print("Precision : ", precision)
print("Recall : ", recall)
print("NPV : ", NPV)
print("FDR : ", FDR)
print("FPR : ", FPR)
print("F1 : ", F1)
print("F2 : ", F2)

# SGD PART ( NOT WORKING RUNS FOREVER )
for i in range(1000):

    for i in range(len(train_labels)):
        y_pred = sigmoid((train_features.dot(weights)) + bias)
        dw += (1 / len(train_labels)) * np.dot(train_features.T, (y_pred - train_labels))
        db = (1 / len(train_labels)) * np.sum(y_pred - train_labels)
        new_dw = [dw[0][0], dw[1][0], dw[2][0]]
        count = 0
        if (i % 100 == 0 and i != 0) or i == len(train_labels) - 1:
            while count < 3:
                weights[count] = weights[count] - new_dw[count] * lr[3]
                count = count + 1
                bias = bias - lr[3] * db
                dw = 0
                db = 0
print(weights)

predictions_2 = []
for i in range(len(test_features)):
    y_pred = 0
    z = weights[0] * test_features[i][0] + weights[1] * test_features[i][1] + weights[2] * test_features[i][2] + bias
    y_pred = sigmoid(z)
    print(y_pred)
    if y_pred > 0.5:
        Y = 1
    else:
        Y = 0
    predictions_2.append(Y)
print(predictions_2)
