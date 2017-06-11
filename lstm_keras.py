import numpy as np
import pickle
import sys
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

hidden_states = 3
input_dim = 7
timesteps = 100

model = Sequential()
model.add(LSTM(hidden_states, input_shape=(timesteps, input_dim)))

w_i = np.arange(input_dim * hidden_states).reshape(input_dim,hidden_states) * -0.05 + 0.00
w_f = np.arange(input_dim * hidden_states).reshape(input_dim,hidden_states) * 0.05 + 0.05
w_c = np.arange(input_dim * hidden_states).reshape(input_dim,hidden_states) * -0.05 + 0.02
w_o = np.arange(input_dim * hidden_states).reshape(input_dim,hidden_states) * 0.05 + 0.03

u_i = np.arange(hidden_states * hidden_states).reshape(hidden_states,hidden_states) * -0.05 + 0.04
u_f = np.arange(hidden_states * hidden_states).reshape(hidden_states,hidden_states) * 0.05 + 0.05
u_c = np.arange(hidden_states * hidden_states).reshape(hidden_states,hidden_states) * -0.05 + 0.06
u_o = np.arange(hidden_states * hidden_states).reshape(hidden_states,hidden_states) * 0.05 + 0.07

b_i = np.arange(hidden_states) * 0.05 + 0.08
b_f = np.arange(hidden_states) * 0.05 + 0.09
b_c = np.arange(hidden_states) * 0.05 + 0.001
b_o = np.arange(hidden_states) * 0.05 + 0.051

u_o = u_o * u_o

w1 = np.concatenate([w_i, w_f, w_c, w_o], axis=1)
w2 = np.concatenate([u_i, u_f, u_c, u_o], axis=1)
w3 = np.concatenate([b_i, b_f, b_c, b_o], axis=0)

print(w1.shape, w2.shape, w3.shape)

model.set_weights([w1, w2, w3])
print(model.get_weights())
print([w.shape for w in model.get_weights()])
h_old = np.zeros(hidden_states)
c_old = np.array(h_old)

pred_input = np.arange(timesteps * input_dim).reshape(1,timesteps,input_dim) * 0.05


def sigmoid(x):
    #x = (x * slope) + shift
    x = (0.2 * x) + 0.5
    x = np.clip(x, 0, 1)
    return x

for t in range(timesteps):
    x = pred_input[0,t,:]

    print("x", x)
    print("hold", h_old)

    i = sigmoid(np.dot(w_i.T, x) + np.dot(u_i, h_old)+b_i)
    c_tilde = np.tanh(np.dot(w_c.T, x)+np.dot(u_c, h_old)+b_c)
    f = sigmoid(np.dot(w_f.T, x) + np.dot(u_f, h_old)+b_f)
    c_new = i * c_tilde + f * c_old
    o = sigmoid(np.dot(w_o.T, x) + np.dot(u_o, h_old)+b_o)
    h_new = o * np.tanh(c_new)

    print("i", i)
    print("C tilde", c_tilde)
    print("f ", f)
    print("C new", c_new)
    print("O", o)
    print("next h", h_new)

    h_old[:] = h_new
    c_old[:] = c_new

print("Expected output", model.predict(pred_input))
