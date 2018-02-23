# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


X= np.array([[1,1],
     [1,0],
    [0,1],
     [0,0]])

Y= np.array([[1],[0],[0],[0]])


input_dim = 2
hidden_dim = 3
output_dim = 1
num_epoch = 100000
learning_rate = 0.01
learning_curve = []
model = {}

def init():
    np.random.seed(1)
    model['W1'] = np.random.randn(input_dim,hidden_dim) / np.sqrt(input_dim)
    model['W2'] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    model['b1'] = np.random.randn(1,hidden_dim)
    model['b2'] = np.random.randn(1,output_dim)
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def calculate_error(y,y_pred):
    return y - y_pred

def accuracy(y_actual, y_pred):
    for i in range(len(y_actual)):
        if (y_pred[i] > 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    cm = confusion_matrix(y_actual, y_pred)
    return (cm[0][0] + cm [1][1])/ len(y_actual)
    
    
def forward_propagation(X_test):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    z1 = X_test.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    return a2

def backward_propagation(X_train, Y_train):
    for i in range(num_epoch):
        W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
        z1 = X_train.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        
        error_a2 = calculate_error(Y_train, a2)
        slope_a2 = sigmoid_derivative(a2)
        slope_a1 = sigmoid_derivative(a1)
        delta_a2 = error_a2 * slope_a2
        error_a1 = delta_a2.dot(W2.T)
        delta_a1 = error_a1 * slope_a1
        
        model['W1'] = W1 + learning_rate * X_train.T.dot(delta_a1)
        model['W2'] = W2 + learning_rate * (a1.T.dot(delta_a2))
        model['b1'] = b1 + np.sum(delta_a1, axis=0, keepdims=True)
        model['b2'] = b2 + np.sum(delta_a2, axis=0, keepdims=True)
        learning_curve.append(np.mean(error_a2))
        if(i%1000 == 0):
            print (np.mean(error_a2))
        
init()
akurasi_before_backprop = accuracy(Y, forward_propagation(X))
backward_propagation(X, Y)
akurasi_after_backprop = accuracy(Y, forward_propagation(X))


learning_curve = np.array(learning_curve, dtype=np.float32)
plt.plot(learning_curve)
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

