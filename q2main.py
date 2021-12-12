import pandas as pd
import numpy as np
import csv
import sympy as syn
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print("hi")
features =[]
count = 0
with open('question-2-features.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            features.append(row)
            #print(arr)
features =np.array(features)
features = features.astype(np.float64)


labels =[]
count = 0
with open('question-2-labels.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            labels.append(row)

            #print(arr)

labels =np.array(labels)
labels = labels.astype(np.float64)
#print(labels)

print(np.linalg.matrix_rank(features.dot(features.T)))

x1=[]
for i in range(len(features)):
    x1.append(features[i][12])
x1 =np.array(x1)
total = 0
total_y =0
mse_l = 0
x11 = x1
bias = np.ones((len(x11),1))
x11 = np.reshape(x11,(len((x11)),1))
final_matrix = np.append(bias,x11,axis=1)
final_t = final_matrix.T
dot_product = final_t.dot(final_matrix)
dot_product = np.linalg.inv(dot_product)
hello = final_t.dot(labels)
weights_l= dot_product.dot(hello)
w0=weights_l[0]
w1=weights_l[1]
print("weight 0 : ", weights_l[0],"weight 1 : ", weights_l[1])
y_pred = []
for i in range(len(labels)):
    equation = w0 + w1*x1[i]
    y_pred.append(equation)
    mse_l += (labels[i] - equation) ** 2
mse_l = mse_l/len(labels)
print("MSE FOR 2.3 :", mse_l)
y_pred = np.array(y_pred)
plt.scatter(x1, labels, color='red')
plt.plot(x1, y_pred)
plt.show()



x11 = x1
x12 = x1*x1
bias = np.ones((len(x11),1))
x11 = np.reshape(x11,(len((x11)),1))
x12 = np.reshape(x12,(len((x11)),1))
final_matrix = np.append(bias,x11,axis=1)
final_matrix = np.append(final_matrix,x12,axis=1)
#print(final_matrix)
final_t = final_matrix.T
#print(final_t)
dot_product = final_t.dot(final_matrix)
#print(dot_product)
dot_product = np.linalg.inv(dot_product)
hello = final_t.dot(labels)
weights = dot_product.dot(hello)


print("weight 0 : ", weights[0],"weight 1 : ", weights[1],"weight 2 : ", weights[2])
y_pred_poly=[]
mse_p = 0
for i in range(len(x1)):
    equation = weights[0] + weights[1]*x1[i] + weights[2]*(x1[i]**2)
    y_pred_poly.append(equation)
    mse_p += (labels[i] - equation) ** 2
mse_p = mse_p/len(labels)
print("MSE FOR 2.4 :", mse_p)

y_pred_poly = np.array(y_pred_poly)
plt.scatter(x1, labels, color='green')
plt.scatter(x1, y_pred_poly)
plt.show()


