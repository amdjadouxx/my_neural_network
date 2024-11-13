import math
import random

def exp(x):
    return math.exp(x)

def tanh(x):
    return math.tanh(x)

def heaviside(x, value):
    return 1 if x >= 0 else 0

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def random_binomial(n, p, size):
    return [1 if random.random() < p else 0 for _ in range(size)]

def random_array(shape):
    return [[random.random() - 0.5 for _ in range(shape[1])] for _ in range(shape[0])]

def argmax(arr):
    return max(range(len(arr)), key=lambda i: arr[i])