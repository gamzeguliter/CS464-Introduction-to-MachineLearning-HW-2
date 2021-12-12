import csv
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    print("hi")
count = 0
df= []
with open('images.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            arr = np.array(row)
            arr = arr.astype(np.int32)
            df.append(arr)

df =np.array(df)
i = 0
#print(df)
count_row = 0
count_col = 0

df = (df - df.mean()) / df.std(ddof=0)
covariance_df = np.cov(df.T)
#print(df)
#print(covariance_df)
eigen_values, eigen_vectors = np.linalg.eig(covariance_df)
eigen_values_sorted = np.sort(eigen_values)[::-1]
#print(eigen_values_sorted)

eigen_vectors_sorted = np.sort(eigen_vectors)[::-1]
top_10_eigen = eigen_values_sorted[0:10]
top_10_eigen_vec = eigen_vectors_sorted[0:10]
fig = plt.figure(figsize=(10, 10))

for i in range(10):
    fig.add_subplot(2, 5, i+1)
    plt.imshow(eigen_vectors[i].reshape(48,48),cmap="gray")
plt.show()

PVE = []
for i in top_10_eigen:
    PVE.append((i / sum(top_10_eigen)) * 100)
#print(PVE)

# creating the bar plot

fig = plt.figure(figsize=(10, 5))
x = [1,2,3,4,5,6,7,8,9,10]
plt.bar(x, PVE, color='maroon',
        width=0.4)
plt.xlabel("PCA's")
plt.ylabel("PVE")
plt.show()

# reconstructing eigenfaces
fig = plt.figure(figsize=(10, 10))
k_array = [1, 10, 50, 100, 500,2000]
for i in range(6):
    k = k_array[i]
    temp= np.zeros((len(eigen_vectors),k),dtype = float)
    temp= eigen_vectors[:,0:k]
    z= df.dot(temp)
    temp2 = z.dot(np.transpose(temp))
    fig.add_subplot(2, 3, i+1)
    plt.imshow(temp2[0].reshape(48,48),cmap="gray")
plt.show()