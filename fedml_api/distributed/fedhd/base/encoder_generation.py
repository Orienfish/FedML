import pickle
import numpy as np
import math


# base vector, matrix 1x1000 1000x(single sample)
base_matrix = []
input_length = 2048
D = 5000

mu = 0.0
sigma = 1.0
for i in range(0, D):
    base_matrix.append(np.random.normal(mu, sigma, input_length))
base_vector = np.random.uniform(0, 2*math.pi, D)

print("Input Length: " + str(input_length))
print("Base vector: ")
print(base_vector.shape)
print("Base metrix: ")
print(len(base_matrix))
print(len(base_matrix[0]))
# global, all client are the same

with open('base_matrix_cifar10_6_26.hd', 'wb') as file:
  pickle.dump(base_matrix, file)
with open('base_vector_cifar10_6_26.hd', 'wb') as file:
  pickle.dump(base_vector, file)