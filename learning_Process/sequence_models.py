import  numpy as np
#from utils import *
import random
import copy
import tensorflow as tf
np.random.seed(0)
import pandas as pd
import re
import logging
import torch
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, BatchNormalization, TimeDistributed, Conv1D, Dense, Dropout, LSTM, Input, Masking, Input, Activation, Reshape, Lambda, RepeatVector, Embedding, Bidirectional, Concatenate, Dot, Multiply, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from  sklearn.decomposition import PCA
from tensorflow.keras.models import model_from_json
from transformers import DistilBertTokenizerFast #, TFDistilBertModel
from transformers import TFDistilBertForTokenClassification
from transformers import TFDistilBertForQuestionAnswering
from transformers import DistilBertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
np.random.seed(1)
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date
from babel.dates import format_date
from pydub import AudioSegment
#from nmt_utils import *
#from td_utils import *
import matplotlib.pyplot as plt
#from shakespeare_utils import *
import sys
import os
import glob
import io
import emoji
#from emo_utils import *
#from music21 import *
#from grammar import *
#from qa import *
#from preprocess import * 
#from w2v_utils import *
#from music_utils import *
#from data_utils import *
#from outputs import *
#from test_utils import *

'''from rnn_utils import *
from public_tests import *
def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    # Retrieve parameters from "parameters"

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by =  parameters["by"]

    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)

# UNIT TESTS
rnn_cell_forward_tests(rnn_cell_forward)

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """

    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    # initialize "a" and "y_pred" with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    # Initialize a_next
    a_next = a0
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        xt = x[:, :, t]
        a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the prediction in y
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches"
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)
    return a, y_pred, caches

np.random.seed(1)
x_tmp = np.random.randn(3, 10, 4)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5, 5)
parameters_tmp['Wax'] = np.random.randn(5, 3)
parameters_tmp['Wya'] = np.random.randn(2, 5)
parameters_tmp['ba'] = np.random.randn(5, 1)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))

#UNIT TEST    
rnn_forward_test(rnn_forward)

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """
    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev, xt), axis=0) 
    ft = sigmoid(np.dot(Wf, concat) + bf)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    it = sigmoid(np.dot(Wi, concat) + bi)
    c_next = it * cct + ft * c_prev
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
c_prev_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi'] = np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)

print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))

# UNIT TEST
lstm_cell_forward_test(lstm_cell_forward)

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        x_t = x[:, :, t]
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x_t, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, c, caches

np.random.seed(1)
x_tmp = np.random.randn(3, 10, 7)
a0_tmp = np.random.randn(5, 10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5 + 3)
parameters_tmp['bf'] = np.random.randn(5, 1)
parameters_tmp['Wi'] = np.random.randn(5, 5 + 3)
parameters_tmp['bi']= np.random.randn(5, 1)
parameters_tmp['Wo'] = np.random.randn(5, 5 + 3)
parameters_tmp['bo'] = np.random.randn(5, 1)
parameters_tmp['Wc'] = np.random.randn(5, 5 + 3)
parameters_tmp['bc'] = np.random.randn(5, 1)
parameters_tmp['Wy'] = np.random.randn(2, 5)
parameters_tmp['by'] = np.random.randn(2, 1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))

# UNIT TEST    
lstm_forward_test(lstm_forward)
data = open("dinos.txt", "r").read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size =  len(data) , len(chars)
print(f"there are {data_size} total characters and {vocab_size} unique characters in your data. ")
chars = sorted(chars)
print(chars)
char_to_ix = {ch:i for i, ch  in enumerate(chars)}
ix_to_char = dict(enumerate(chars))
print(ix_to_char)

def clip(gradients, maxvalue):
    """
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    """
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["db"], gradients["dby"]
    # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby].
    for gradient in gradients:
        np.clip(gradients[gradient], -maxvalue, maxvalue, out=gradients[gradient])

    return {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

# Test with a max value of 10
def clip_test(target, mValue):
    print(f"\nGradients for mValue={mValue}")
    np.random.seed(3)
    dWax = np.random.randn(5, 3) * 10
    dWaa = np.random.randn(5, 5) * 10
    dWya = np.random.randn(2, 5) * 10
    db = np.random.randn(5, 1) * 10
    dby = np.random.randn(2, 1) * 10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

    gradients2 = target(gradients, mValue)
    print("gradients[\"dWaa\"][1][2] =", gradients2["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients2["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients2["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients2["db"][4])
    print("gradients[\"dby\"][1] =", gradients2["dby"][1])
    
    for grad in gradients2.keys():
        valuei = gradients[grad]
        valuef = gradients2[grad]
        mink = np.min(valuef)
        maxk = np.max(valuef)
        assert mink >= -abs(mValue), f"Problem with {grad}. Set a_min to -mValue in the np.clip call"
        assert maxk <= abs(mValue), f"Problem with {grad}.Set a_max to mValue in the np.clip call"
        index_not_clipped = np.logical_and(valuei <= mValue, valuei >= -mValue)
        assert np.all(valuei[index_not_clipped] == valuef[index_not_clipped]), f" Problem with {grad}. Some values that should not have changed, changed during the clipping process."
    
    print("\033[92mAll tests passed!\x1b[0m")
    
clip_test(clip, 10)
clip_test(clip, 5)

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- Python dictionary mapping each character to an index.
    seed -- seed is seed

    Returns:
    indices -- A list of length n containing the indices of the sampled characters.
    """
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, b, by = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["b"], parameters["by"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate
    indices = []
    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    # Initialize idx to -1
    idx = -1
    # Loop over time-steps t. At each time-step:
    # Sample a character from a probability distribution 
    # And append its index (`idx`) to the list "indices". 
    # You'll stop if you reach 50 characters 
    # (which should be very unlikely with a well-trained model).
    # Setting the maximum number of characters helps with debugging and prevents infinite loops. 
    counter = 0
    newline_character = char_to_ix["\n"]

    while(idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        np.random.seed(counter + seed) 
        y_1d = y.ravel()
        idx = np.random.choice(range(len(y_1d)), p = y_1d)
        # Append the index to "indices"
        indices.append(idx)
        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        # Update "a_prev" to be "a"
        a_prev =  a
        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix["\n"])

    return indices

def sample_test(target):
    np.random.seed(24)
    _, n_a = 20, 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


    indices = target(parameters, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:\n", indices)
    print("list of sampled characters:\n", [ix_to_char[i] for i in indices])
    
    assert len(indices) < 52, "Indices lenght must be smaller than 52"
    assert indices[-1] == char_to_ix['\n'], "All samples must end with \\n"
    assert min(indices) >= 0 and max(indices) < len(char_to_ix), f"Sampled indexes must be between 0 and len(char_to_ix)={len(char_to_ix)}"
    assert np.allclose(indices[0:6], [23, 16, 26, 26, 24, 3]), "Wrong values"
    
    print("\033[92mAll tests passed!")

sample_test(sample)

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    # Backpropagate through time 
    gradients, a = rnn_backward(X, Y, parameters, cache)
    # Clip your gradients between -5 (min) and 5 (max)
    gradients = clip(gradients, 5)
    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]

def optimize_test(target):
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    X = [12, 3, 5, 11, 22, 3]
    Y = [4, 14, 11, 22, 25, 26]
    old_parameters = copy.deepcopy(parameters)
    loss, gradients, a_last = target(X, Y, a_prev, parameters, learning_rate = 0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    print("a_last[4] =", a_last[4])
    
    assert np.isclose(loss, 126.5039757), "Problems with the call of the rnn_forward function"
    for grad in gradients.values():
        assert np.min(grad) >= -5, "Problems in the clip function call"
        assert np.max(grad) <= 5, "Problems in the clip function call"
    assert np.allclose(gradients['dWaa'][1, 2], 0.1947093), "Unexpected gradients. Check the rnn_backward call"
    assert np.allclose(gradients['dWya'][1, 2], -0.007773876), "Unexpected gradients. Check the rnn_backward call"
    assert not np.allclose(parameters['Wya'], old_parameters['Wya']), "parameters were not updated"
    
    print("\033[92mAll tests passed!")

optimize_test(optimize)

def model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    examples = [x.strip() for x in data_x]
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # for grading purposes
    last_dino_name = "abc"
    
    # Optimization loop
    for j in range(num_iterations):
        
        ### START CODE HERE ###
        
        # Set the index `idx` (see instructions above)
        idx = j%len(examples)
        
        # Set the input X (see instructions above)
        single_example_chars = examples[idx]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]

        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        X = [None] + single_example_ix
        
        # Set the labels Y (see instructions above)
        # The goal is to train the RNN to predict the next letter in the name
        # So the labels are the list of characters that are one time-step ahead of the characters in the input X
        Y = X[1:] 
        # The RNN should predict a newline at the last letter, so add ix_newline to the end of the labels
        ix_newline = [char_to_ix["\n"]]
        Y = Y + ix_newline

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        
        ### END CODE HERE ###
        
        # debug statements to aid in correctly forming X, Y
        if verbose and j in [0, len(examples) -1, len(examples)]:
            print("j = " , j, "idx = ", idx,) 
        if verbose and j in [0]:
            #print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 1000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 1000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))
                
                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
      
            print('\n')
        
    return parameters, last_dino_name

parameters, last_name = model(data.split("\n"), ix_to_char, char_to_ix, 35000, verbose = True)

print("\033[92mAll tests passed!")

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

generate_output()

X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid')
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('shape of X:', X.shape)
print('Shape of Y:', Y.shape)
print('Number of chords', len(chords))
# number of dimensions for the hidden state of each LSTM cell.
n_a = 64
n_values = 90 # number of music values
reshaper = Reshape((1, n_values))                  # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D

def djmodel(Tx, LSTM_cell, densor, reshaper):
    """
    Implement the djmodel composed of Tx LSTM cells where each cell is responsible
    for learning the following note based on the previous note and context.
    Each cell has the following schema: 
            [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
    Arguments:
        Tx -- length of the sequences in the corpus
        LSTM_cell -- LSTM layer instance
        densor -- Dense layer instance
        reshaper -- Reshape layer instance
    
    Returns:
        model -- a keras instance model with inputs [X, a0, c0]
    """
    # Get the shape of input values
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a, ), name="a0")
    c0 = Input(shape=(n_a, ), name="c0")
    a=a0
    c=c0
    # Step 1: Create empty list to append the outputs while you iterate 
    outputs = [] 
    
    # Step 2: Loop from 0 to Tx
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. 
        x = X[:,t,:]
        # Step 2.B: Use reshaper to reshape x to be (1, n_values) 
        x = reshaper(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    return Model(inputs=[X, a0, c0], outputs=outputs)

model = djmodel(Tx=30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)   
output = summary(model) 
comparator(output, djmodel_out) 
# Check your model
model.summary()
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

history = model.fit([X, a0, c0], list(Y), epochs=100, verbose=0)

print(f"loss at epoch 1: {history.history["loss"][0]}")
print(f"loss at epoch 100: {history.history["loss"][99]}")
plt.plot(history.history["loss"])

model.save("my_jazz_model.h5")
model.save_weights('my_jazz_model_weights.h5')
print("Models saved successfully!")

def music_inference_model(LSTM_cell, densor, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Get the shape of input values
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Step 1: Create an empty list of "outputs" to later store your predicted values 
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Step 2.A: Perform one step of LSTM_cell. Use "x", not "x0" 
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell 
        out = densor(a)
        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 90) 
        outputs.append(out)

        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = tf.math.argmax(out, axis=-1)
        x = tf.one_hot(x, n_values)
        # Step 2.E: 
        # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
        x = RepeatVector(1)(x)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" 
    return Model(inputs=[x0, a0, c0], outputs=outputs)

inference_model = music_inference_model(LSTM_cell, densor, Ty = 50)

inference_summary = summary(inference_model) 
comparator(inference_summary, music_inference_model_out)

inference_model.summary()

x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 90), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 90), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    n_values = x_initializer.shape[2]

    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])

    indices = np.argmax(pred, axis=-1)

    results = to_categorical(indices, num_classes=n_values)

    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    # Special case. Consider the case u = [0, 0], v=[0, 0]
    if np.all(u == v):
        return 1
    dot = np.dot(u,v)
    # Compute the L2 norm of u 
    norm_u = np.sqrt(np.sum(u * u))
    # Compute the L2 norm of v 
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) 
    return dot / (norm_u * norm_v)

father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
# END SKIP FOR GRADING

# PUBLIC TESTS
def cosine_similarity_test(target):
    a = np.random.uniform(-10, 10, 10)
    b = np.random.uniform(-10, 10, 10)
    c = np.random.uniform(-1, 1, 23)
        
    assert np.isclose(cosine_similarity(a, a), 1), "cosine_similarity(a, a) must be 1"
    assert np.isclose(cosine_similarity((c >= 0) * 1, (c < 0) * 1), 0), "cosine_similarity(a, not(a)) must be 0"
    assert np.isclose(cosine_similarity(a, -a), -1), "cosine_similarity(a, -a) must be -1"
    assert np.isclose(cosine_similarity(a, b), cosine_similarity(a * 2, b * 4)), "cosine_similarity must be scale-independent. You must divide by the product of the norms of each input"

    print("\033[92mAll test passed!")
    
cosine_similarity_test(cosine_similarity)

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    # convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    # Get the word embeddings e_a, e_b and e_c
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    max_cosine_sim = -100         # Initialize max_cosine_sim to a large negative number
    best_word = None              # Initialize best_word with None, it will help keep track of the word to output
    # loop over the whole word vector set
    for word, vec in word_to_vec_map.items():
        # to avoid best_word being one the input words, skip the input word_c
        # skip word_c from query
        if word == word_c:
            continue
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity(e_b - e_a, vec - e_c)
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word

# PUBLIC TEST
def complete_analogy_test(target):
    a = [3, 3] # Center at a
    a_nw = [2, 4] # North-West oriented vector from a
    a_s = [3, 2] # South oriented vector from a
    
    c = [-2, 1] # Center at c
    # Create a controlled word to vec map
    word_to_vec_map = {'a': a,
                       'synonym_of_a': a,
                       'a_nw': a_nw, 
                       'a_s': a_s, 
                       'c': c, 
                       'c_n': [-2, 2], # N
                       'c_ne': [-1, 2], # NE
                       'c_e': [-1, 1], # E
                       'c_se': [-1, 0], # SE
                       'c_s': [-2, 0], # S
                       'c_sw': [-3, 0], # SW
                       'c_w': [-3, 1], # W
                       'c_nw': [-3, 2] # NW
                      }
    
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
            
    assert(target('a', 'a_nw', 'c', word_to_vec_map) == 'c_nw')
    assert(target('a', 'a_s', 'c', word_to_vec_map) == 'c_s')
    assert(target('a', 'synonym_of_a', 'c', word_to_vec_map) != 'c'), "Best word cannot be input query"
    assert(target('a', 'c', 'a', word_to_vec_map) == 'c')

    print("\033[92mAll tests passed")
    
complete_analogy_test(complete_analogy) 

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large'), ]

#for triad in triads_to_try:
#    print(f"{triad[0]} -> {triad[1]} :: {triad[2]} -> {complete_analogy(*triad, word_to_vec_map)}")

g = word_to_vec_map["woman"] - word_to_vec_map["man"]
print(g)

print ('List of names and their similarities with constructed vector:')

name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))

def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    e = word_to_vec_map[word]
    e_bias = ( np.dot(e, g) / np.sum(g ** 2) ) * g
    return e - e_bias

e = "receptionist"
print(f"cosine similarity between {e} and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g),)

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print(f"cosine similarity between {e} and g, after neutralizing: ",cosine_similarity(e_debiased, g),)

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    word1, word2 = pair
    ew1, ew2 = word_to_vec_map[word1], word_to_vec_map[word2]
    meu = (ew1 + ew2) / 2
    meu_B = ( np.dot(meu, bias_axis) / np.linalg.norm(bias_axis ** 2) ) * bias_axis
    meu_ortho = meu - meu_B
    ew1B = ( np.dot(ew1, bias_axis) / np.linalg.norm(bias_axis ** 2) ) * bias_axis
    ew2B = ( np.dot(ew2, bias_axis) / np.linalg.norm(bias_axis ** 2) ) * bias_axis
    ew1B_corrected = np.sqrt(np.abs(1 - np.linalg.norm(meu_ortho) ** 2) )* ( (ew1B - meu_B) / np.linalg.norm((ew1 - meu_ortho) - meu_B) )
    ew2B_corrected = np.sqrt(np.abs(1 - np.linalg.norm(meu_ortho) ** 2) )* ( (ew2B - meu_B) / np.linalg.norm((ew1 - meu_ortho) - meu_B) )
    e1 = ew1B_corrected + meu_ortho
    e2 = ew2B_corrected + meu_ortho
    return e1, e2

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=len).split())

for idx in range(10):
    print(X_train[idx], label_to_emoji(Y_train[idx]))

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

idx = 30
print(f"Sentence '{X_train[idx]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

word = "cucumber"
idx = 289846
print(f"the index of {word} in the vocabulary is {word_to_index[word]}")
print(f"the {idx}th word in the dictionary is {index_to_word[idx]} ")

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
    """
    # Get a valid word contained in the word_to_vec_map.
    any_word = list(word_to_vec_map.keys())[0]

    words = sentence.lower().split()
    any_word_vector = word_to_vec_map[any_word]
    avg = np.zeros(shape=any_word_vector.shape)
    count = 0

    for w in words:
        if w in list(word_to_vec_map.keys()):
            avg += word_to_vec_map[w]
            count += 1
    if count > 0:
        avg /= count

    return avg

avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = \n", avg)

def sentence_to_avg_test(target):
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
        
    avg = target("a a_nw c_w a_s", word_to_vec_map)
    assert tuple(avg.shape) == tuple(word_to_vec_map['a'].shape),  "Check the shape of your avg array"  
    assert np.allclose(avg, [1.25, 2.5]),  "Check that you are finding the 4 words"
    avg = target("love a a_nw c_w a_s", word_to_vec_map)
    assert np.allclose(avg, [1.25, 2.5]), "Divide by count, not len(words)"
    avg = target("love", word_to_vec_map)
    assert np.allclose(avg, [0, 0]), "Average of no words must give an array of zeros"
    avg = target("c_se foo a a_nw c_w a_s deeplearning c_nw", word_to_vec_map)
    assert np.allclose(avg, [0.1666667, 2.0]), "Debug the last example"
    
    print("\033[92mAll tests passed!")
    
sentence_to_avg_test(sentence_to_avg)

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 200):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    any_word = list(word_to_vec_map.keys())[0]

    cost = 0
    # Define number of training examples
    m = Y.shape[0]                        # number of training examples
    n_y = len(np.unique(Y))               # number of classes  
    n_h = (word_to_vec_map[any_word].shape[0])      # dimensions of the GloVe vectors 
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y, ))
    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y)
    # Optimization loop
    for t in range(num_iterations):    # Loop over the number of iterations
        for i in range(m):             # Loop over the training examples
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)
            # Forward propagate the avg through the softmax layer
            z = np.add(np.dot(W, avg), b)
            a = softmax(z)
            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = -np.sum(np.dot(Y_oh[i], np.log(a)))
            #  Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape((n_y, 1)), avg.reshape((1, n_h)))
            db = dz
            # Update parameters with Stochastic Gradient Descent
            W -= (learning_rate * dW)
            b -= (learning_rate * db)
        
        if t % 10 == 0:
            print(f"Epoch: {t} -- cost: {cost}")
            pred = predict(X, Y, W, b, word_to_vec_map)   #predict is defined in emo_utils.py

    return pred, W, b

def model_test(target):
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
        
    # Training set. Sentences composed of a_* words will be of class 0 and sentences composed of c_* words will be of class 1
    X = np.asarray(['a a_s synonym_of_a a_n c_sw', 'a a_s a_n c_sw', 'a_s  a a_n', 'synonym_of_a a a_s a_n c_sw', " a_s a_n",
                    " a a_s a_n c ", " a_n  a c c c_e",
                   'c c_nw c_n c c_ne', 'c_e c c_se c_s', 'c_nw c a_s c_e c_e', 'c_e a_nw c_sw', 'c_sw c c_ne c_ne'])
    
    Y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    np.random.seed(10)
    pred, W, b = model(X, Y, word_to_vec_map, 0.0025, 110)
    
    assert W.shape == (2, 2), "W must be of shape 2 x 2"
    assert np.allclose(pred.transpose(), Y), "Model must give a perfect accuracy"
    assert np.allclose(b[0], -1 * b[1]), "b should be symmetric in this example"
    
    print("\033[92mAll tests passed!")
    
model_test(model)

#np.random.seed(0)
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i cherish you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy", "Let's go gambling"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)
print(Y_test.shape)
print(f'           {label_to_emoji(0)}    {label_to_emoji(1)}    {label_to_emoji(2)}    {label_to_emoji(3)}   {label_to_emoji(4)}')
print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=["actual"], colnames=["predict"], margins=True))
plot_confusion_matrix(Y_test, pred_test)

for idx, val in enumerate(["I", "Like", "Learning"]):
    print(idx, val)

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]              # number of training 
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros([m, max_len])

    for i in range(m):
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            # if w exists in the word_to_index dictionary
            if w in word_to_index:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j += 1
        
    return X_indices

def sentences_to_indices_test(target):
    
    # Create a word_to_index dictionary
    word_to_index = {}
    for idx, val in enumerate(["i", "like", "learning", "deep", "machine", "love", "smile", '´0.=']):
        word_to_index[val] = idx + 1;
       
    max_len = 4
    sentences = np.array(["I like deep learning", "deep ´0.= love machine", "machine learning smile"]);
    indexes = target(sentences, word_to_index, max_len)
    print(indexes)
    
    assert type(indexes) == np.ndarray, "Wrong type. Use np arrays in the function"
    assert indexes.shape == (sentences.shape[0], max_len), "Wrong shape of ouput matrix"
    assert np.allclose(indexes, [[1, 2, 4, 3],
                                 [4, 8, 6, 5],
                                 [5, 3, 7, 0]]), "Wrong values. Debug with the given examples"
    
    print("\033[92mAll tests passed!")
    
sentences_to_indices_test(sentences_to_indices)

X1 = np.array(["funny lol", "let's go gambling", "food is ready for you"])
X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
print(f"X1 = {X1}")
print(f"X1_indices =\n  {X1_indices}")

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a tensorflow Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer tensorflow instance
    """

    vocab_size = len(word_to_index) + 1
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros([vocab_size, emb_dim])
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx] = word_to_vec_map[word]
    # Define tensorflow embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_size, emb_dim, trainable=False)
    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None, ))
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def pretrained_embedding_layer_test(target):
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
        
    # Create a word_to_index dictionary
    word_to_index = {}
    for idx, val in enumerate(list(word_to_vec_map.keys())):
        word_to_index[val] = idx;
        
    np.random.seed(1)
    embedding_layer = target(word_to_vec_map, word_to_index)
    
    assert type(embedding_layer) == Embedding, "Wrong type"
    assert embedding_layer.input_dim == len(list(word_to_vec_map.keys())) + 1, "Wrong input shape"
    assert embedding_layer.output_dim == len(word_to_vec_map['a']), "Wrong output shape"
    assert np.allclose(embedding_layer.get_weights(), 
                       [[[ 3, 3], [ 3, 3], [ 2, 4], [ 3, 2], [ 3, 4],
                       [-2, 1], [-2, 2], [-1, 2], [-1, 1], [-1, 0],
                       [-2, 0], [-3, 0], [-3, 1], [-3, 2], [ 0, 0]]]), "Wrong vaulues"
    print("\033[92mAll tests passed!")
       
    
pretrained_embedding_layer_test(pretrained_embedding_layer)

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print(f"weight[0][1][1] {embedding_layer.get_weights()[0][1][1]}")
print(f"input dims: {embedding_layer.input_dim}")
print(f"output dims: {embedding_layer.output_dim}")

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in tensorflow
    """
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = Input(shape=input_shape, dtype="int32")
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # Propagate sentence_indices through your embedding layer
    embedding = embedding_layer(sentence_indices)
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences, So, set return_sequences = True
    # If return_sequences = False, the LSTM returns only tht last output in output sequence
    X = LSTM(units=128, return_sequences=True)(embedding)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(units=5)(X)
    # Add a softmax activation
    X = Activation("softmax")(X)
    # Create Model instance which converts sentence_indices into X.
    return Model(inputs=sentence_indices, outputs=X)

def Emojify_V2_test(target):
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
        
    # Create a word_to_index dictionary
    word_to_index = {}
    for idx, val in enumerate(list(word_to_vec_map.keys())):
        word_to_index[val] = idx;
        
    maxLen = 4
    model = target((maxLen,), word_to_vec_map, word_to_index)
    
    expectedModel = [['InputLayer', [(None, 4)], 0], ['Embedding', (None, 4, 2), 30], ['LSTM', (None, 4, 128), 67072, (None, 4, 2), 'tanh', True], ['Dropout', (None, 4, 128), 0, 0.5], ['LSTM', (None, 128), 131584, (None, 4, 128), 'tanh', False], ['Dropout', (None, 128), 0, 0.5], ['Dense', (None, 5), 645, 'linear'], ['Activation', (None, 5), 0]]
    comparator(summary(model), expectedModel)
    
    
#Emojify_V2_test(Emojify_V2)
model = Emojify_V2((maxLen, ), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print(f"Test Accuracy: {acc}")
C = 5
Y_oh_test = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)

for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if num != Y_test[i]:
        print(f'Expected emoji:{label_to_emoji(Y_test[i])} prediction: {X_test[i]}{label_to_emoji(num).strip()}')

X_test = np.array(["The movie is not good and not enjoyable"])
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
print(f'{X_test[0]} {label_to_emoji(np.argmax(model.predict(X_test_indices)))}')

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print(dataset[:10])
    
Tx = 30
Ty = 10

X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

print()

index = 3
print(f"source date: {dataset[index][0]}")
print(f"target date: {dataset[index][1]}")
print()
print(f"source after preprocessing(indices): {X[index]}")
print(f"target after preprocessing(indices): {Y[index]}")
print()
print(f"source after preprocessing(one-hot-vector): {Xoh[index]}")
print(f"target after preprocessing(one-hot-vector): {Yoh[index]}")
# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatinator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(softmax, name="attention_weights")
dotor = Dot(axes=1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" 
    s_prev = repeator(s_prev) 
    # Use concatenator to concatenate a and s_prev on the last axis 
    concat = concatinator([a,s_prev]) 
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. 
    e = densor1(concat) 
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. 
    energies = densor2(e) 
    # Use "activator" on "energies" to compute the attention weights "alphas" 
    alphas = activator(energies) 
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell 
    return dotor([alphas, a])

def one_step_attention_test(target):

    m = 10
    Tx = 30
    n_a = 32
    n_s = 64
    #np.random.seed(10)
    a = np.random.uniform(1, 0, (m, Tx, 2 * n_a)).astype(np.float32)
    s_prev =np.random.uniform(1, 0, (m, n_s)).astype(np.float32) * 1
    context = target(a, s_prev)
    
    assert isinstance(context, tf.Tensor), "Unexpected type. It should be a Tensor"
    assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
    assert np.all(context.numpy() > 0), "All output values must be > 0 in this example"
    assert np.all(context.numpy() < 1), "All output values must be < 1 in this example"

    #assert np.allclose(context[0][0][0:5].numpy(), [0.50877404, 0.57160693, 0.45448175, 0.50074816, 0.53651875]), "Unexpected values in the result"
    print("\033[92mAll tests passed!")
    
one_step_attention_test(one_step_attention)

n_a = 32     # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64     # number of units for the post-attention LSTM's hidden state "s"
# Please note, this is the post attention LSTM cell. 
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation="softmax")

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name="s0")
    c0 = Input(shape=(n_s,), name="c0")
    s = s0
    c = c0
    # Initialize empty list of outputs
    outputs = []
    # Step 1: Define your pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a * 2))(X)
    # Step 2: Iterate for Ty steps
    for y in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a, s)
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM 
        out = output_layer(s)
        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)
    # Step 3: Create model instance taking three inputs and returning the list of outputs.
    return Model(inputs=[X, s0, c0], outputs=outputs)

# UNIT TEST
from test_utils import *

def modelf_test(target):
    m = 10
    Tx = 30
    n_a = 32
    n_s = 64
    len_human_vocab = 37
    len_machine_vocab = 11
    
    
    model = target(Tx, Ty, n_a, n_s, len_human_vocab, len_machine_vocab)
    
    print(summary(model))

    
    expected_summary = [['InputLayer', [(None, 30, 37)], 0],
                         ['InputLayer', [(None, 64)], 0],
                         ['Bidirectional', (None, 30, 64), 17920],
                         ['RepeatVector', (None, 30, 64), 0, 30],
                         ['Concatenate', (None, 30, 128), 0],
                         ['Dense', (None, 30, 10), 1290, 'tanh'],
                         ['Dense', (None, 30, 1), 11, 'relu'],
                         ['Activation', (None, 30, 1), 0],
                         ['Dot', (None, 1, 64), 0],
                         ['InputLayer', [(None, 64)], 0],
                         ['LSTM',[(None, 64), (None, 64), (None, 64)], 33024,[(None, 1, 64), (None, 64), (None, 64)],'tanh'],
                         ['Dense', (None, 11), 715, 'softmax']]

    assert len(model.outputs) == 10, f"Wrong output shape. Expected 10 != {len(model.outputs)}"

    comparator(summary(model), expected_summary)

#modelf_test(modelf)

model = modelf(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

model.summary()

opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"] * 10)

assert opt.learning_rate == 0.005, "Set the lr parameter to 0.005"
assert opt.beta_1 == 0.9, "Set the beta_1 parameter to 0.9"
assert opt.beta_2 == 0.999, "Set the beta_2 parameter to 0.999"
#assert opt.decay == 0.01, "Set the decay parameter to 0.01"
assert model.loss == "categorical_crossentropy", "Wrong loss. Use 'categorical_crossentropy'"
assert model.optimizer == opt, "Use the optimizer that you have instantiated"
#assert model.compiled_metrics._user_metrics[0] == 'accuracy', "set metrics to ['accuracy']"

print("\033[92mAll tests passed!")

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
model.load_weights("models/model.h5")

EXAMPLES = ['2025 Wednesday july 25th', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
s00 = np.zeros((1, n_s))
c00 = np.zeros((1, n_s))

for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=(len(human_vocab))), source))).swapaxes(0, 1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s00, c00])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    print(f"source: {example}")
    print("output:", ''.join(output),"\n")

model.summary()
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);

x = graph_spectrogram("audio_examples/example_train.wav")
_, data = wavfile.read("audio_examples/example_train.wav")
print(f"Time steps in audio recording before spectrogram {data[:, 0]}")
print(f"Time steps in input after spectrogram {x.shape}")
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375  # The number of time steps in the output of our model
activates, negatives, backgrounds = load_raw_audio('./raw_data/')

print(f"background len should be 10,000, since it is a 10 sec clip\n {len(backgrounds[0])}\n")
print(f"activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n {len(activates[0])}\n")
print(f"activate[1] len: different `activate` clips can have different lengths\n {len(activates[1])}\n")

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segement_end = segment_start + segment_ms - 1
    return (segment_start, segement_end)

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    segment_start, segment_end = segment_time
    # Step 1: Initialize overlap as a "False" flag.
    overlap = False
    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap 
    for start, end in previous_segments:
        if segment_start <= end and segment_end >= start:
            overlap = True
            break

    return overlap

def is_overlapping_test(target):
    assert target((670, 1430), []) == False, "Overlap with an empty list must be False"
    assert target((500, 1000), [(100, 499), (1001, 1100)]) == False, "Almost overlap, but still False"
    assert target((750, 1250), [(100, 750), (1001, 1100)]) == True, "Must overlap with the end of first segment"
    assert target((750, 1250), [(300, 600), (1250, 1500)]) == True, "Must overlap with the begining of second segment"
    assert target((750, 1250), [(300, 600), (600, 1500), (1600, 1800)]) == True, "Is contained in second segment"
    print("\033[92m All tests passed!")
    
is_overlapping_test(is_overlapping)
    
overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
print("Overlap 1 = ", overlap1)
print("Overlap 2 = ", overlap2)

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip.
    segment_time = get_random_time_segment(segment_ms)
    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. To avoid an endless loop
    # we retry 5 times
    retries = 5

    while is_overlapping(segment_time, previous_segments) and retries >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retries -= 1
    # if last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Step 3: Append the new segment_time to the list of previous_segments
        previous_segments.append(segment_time)
        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position=segment_time[0])
        print(segment_time)
    
    else:
        print("timedout")
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time

def insert_audio_clip_test(target):
    np.random.seed(5)
    audio_clip, segment_time = target(backgrounds[0], activates[0], [(0, 4400)])
    duration = segment_time[1] - segment_time[0]
    assert segment_time[0] > 4400, "Error: The audio clip is overlaping with the first segment"
    assert duration + 1 == len(activates[0]) , "The segment length must match the audio clip length"
    assert audio_clip != backgrounds[0] , "The audio clip must be different than the pure background"
    
    # Not possible to insert clip into background
    audio_clip, segment_time = target(backgrounds[0], activates[0], [(0, 9999)])
    assert segment_time[0] == 10000 and segment_time[1] == 10000, "Segment must match the out by max-retry mark"
    assert audio_clip == backgrounds[0], "output audio clip must be exactly the same input background"

    print("\033[92m All tests passed!")

insert_audio_clip_test(insert_audio_clip)

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    _, Ty =  y.shape
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000)
    if segment_end_y < Ty:
        # Add 1 to the correct index in the background label (y)
        for i in range(1, 51):
            if segment_end_y + i < Ty:
                y[0][segment_end_y + i] = 1
            
            else:
                break

    return y

import random
def insert_ones_test(target):
    segment_end_y = random.randrange(0, Ty - 50) 
    segment_end_ms = int(segment_end_y * 10000.4) / Ty;    
    arr1 = target(np.zeros((1, Ty)), segment_end_ms)

    assert type(arr1) == np.ndarray, "Wrong type. Output must be a numpy array"
    assert arr1.shape == (1, Ty), "Wrong shape. It must match the input shape"
    assert np.sum(arr1) == 50, "It must insert exactly 50 ones"
    assert arr1[0][segment_end_y - 1] == 0, f"Array at {segment_end_y - 1} must be 0"
    assert arr1[0][segment_end_y] == 0, f"Array at {segment_end_y} must be 1"
    assert arr1[0][segment_end_y + 1] == 1, f"Array at {segment_end_y + 1} must be 1"
    assert arr1[0][segment_end_y + 50] == 1, f"Array at {segment_end_y + 50} must be 1"
    assert arr1[0][segment_end_y + 51] == 0, f"Array at {segment_end_y + 51} must be 0"

    print("\033[92m All tests passed!")
    
insert_ones_test(insert_ones)

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0,:])
print(f"sanity checks: {arr1[0][1333]} {arr1[0][634]} {arr1[0][635]}")

def create_training_example(background, activates, negatives, Ty):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    Ty -- The number of time steps in the output

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    # Make background quieter
    background -= 20
    # Step 1: Initialize y (label vector) of zeros
    y = np.zeros((1, Ty))
    # Step 2: Initialize segment times as empty list
    previous_segments = []
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indicies = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indicies]
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for audio_clip in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, audio_clip, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)
    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indicies = np.random.randint(len(activates), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indicies]
    # Step 4: Loop over randomly selected negative clips and insert in background
    for audio_clip in random_negatives:
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background, audio_clip, previous_segments)

    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)
    # Export new training example
    # file_handle = background.export("train" + ".wav", format="wav")
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x, y

def create_training_example_test(target):
    np.random.seed(18)
    x, y = target(backgrounds[0], activates, negatives, 1375)
    
    assert type(x) == np.ndarray, "Wrong type for x"
    assert type(y) == np.ndarray, "Wrong type for y"
    assert tuple(x.shape) == (101, 5511), "Wrong shape for x"
    assert tuple(y.shape) == (1, 1375), "Wrong shape for y"
    assert np.all(x > 0), "All x values must be higher than 0"
    assert np.all(y >= 0), "All y values must be higher or equal than 0"
    assert np.all(y < 51), "All y values must be smaller than 51"
    assert np.sum(y) % 50 == 0, "Sum of activate marks must be a multiple of 50"
    assert np.isclose(np.linalg.norm(x), 39745552.52075), "Spectrogram is wrong. Check the parameters passed to the insert_audio_clip function"

    print("\033[92m All tests passed!")

create_training_example_test(create_training_example)

np.random.seed(18)
x, y = create_training_example(backgrounds[0], activates, negatives, Ty)
plt.plot(y[0])
np.random.seed(4543)
nsamples = 32
X = []
Y = []

for i in range(nsamples):
    if i % 10 == 0:
        print(i)

    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    X.append(x.swapaxes(0, 1))
    Y.append(y.swapaxes(0, 1))

X = np.array(X)
Y = np.array(Y)
# Save the data for further uses
# np.save("./XY_train/X.npy", X)
# np.save("./XY_train/Y.npy", Y)
# Load the preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

def modelf(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(rate=0.8)(X)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.8)(X)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)

    return Model(inputs=X_input, outputs=X)

from test_utils import *

def modelf_test(target):
    Tx = 5511
    n_freq = 101
    model = target(input_shape = (Tx, n_freq))
    expected_model = [['InputLayer', [(None, 5511, 101)], 0],
                     ['Conv1D', (None, 1375, 196), 297136, 'valid', 'linear', (4,), (15,), 'GlorotUniform'],
                     ['BatchNormalization', (None, 1375, 196), 784],
                     ['Activation', (None, 1375, 196), 0],
                     ['Dropout', (None, 1375, 196), 0, 0.8],
                     ['GRU', (None, 1375, 128), 125184, True],
                     ['Dropout', (None, 1375, 128), 0, 0.8],
                     ['BatchNormalization', (None, 1375, 128), 512],
                     ['GRU', (None, 1375, 128), 99072, True],
                     ['Dropout', (None, 1375, 128), 0, 0.8],
                     ['BatchNormalization', (None, 1375, 128), 512],
                     ['Dropout', (None, 1375, 128), 0, 0.8],
                     ['TimeDistributed', (None, 1375, 1), 129, 'sigmoid']]
    comparator(summary(model), expected_model)
    
    
model = modelf(input_shape=(Tx, n_freq))
#modelf_test(modelf)
model.summary()
with open("./models/model_new8.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("./models/model.h5")'''

'''def get_angles(pos, i, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        i --   Row vector containing the dimension span [[0, 1, 2, ..., M-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    return pos / (10000 ** (2 * (i // 2) / np.float32(d)))

def get_angles_test(target):
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = target(pos_m, dims, d_model)

    assert type(result) == np.ndarray, "You must return a numpy ndarray"
    assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols =  result[:, 0::2]
    odd_cols = result[:,  1::2]
    assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
    limit = (position - 1) / np.power(10000,14.0/16.0)
    assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"

    print("\033[92mAll tests passed")

get_angles_test(get_angles)

# Example
position = 4
d_model = 8
pos_m = np.arange(position)[:, np.newaxis]
dims = np.arange(d_model)[np.newaxis, :]
get_angles(pos_m, dims, d_model)

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
    # -> angle_rads has dim (positions,d)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_test(target):
    position = 8
    d_model = 16

    pos_encoding = target(position, d_model)
    sin_part = pos_encoding[:, :, 0::2]
    cos_part = pos_encoding[:, :, 1::2]

    assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

    ones = sin_part ** 2  +  cos_part ** 2
    assert np.allclose(ones, np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
    
    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part.numpy() < 0] += np.pi
    angs = angs % (2 * np.pi)
    
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)
    
    assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"
 
    print("\033[92mAll tests passed")

    
positional_encoding_test(positional_encoding)

pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)
plt.pcolormesh(pos_encoding[0], cmap="RdBu")
plt.xlabel("d")
plt.xlim((0, 512))
plt.ylabel("Postion")
plt.colorbar()
plt.show()

def create_padding_mask(seq):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        seq -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, 1, m) binary tensor
    """
    seq = tf.cast(tf.math.equal(seq, 0), dtype=tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]

x = tf.constant([[7., 6., 0., 0., 1.], [1., 2., 3., 0., 0.], [0., 0., 0., 4., 5.]])
print(create_padding_mask(x))
print(tf.keras.activations.softmax(x))
print(tf.keras.activations.softmax(x + create_padding_mask(x) * -1.0e9))

def create_look_ahead_mask(size):
    """
    Returns an upper triangular matrix filled with ones
    
    Arguments:
        size -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    return tf.linalg.band_part(tf.ones((size, size)), -1, 0)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

    Arguments:
        q -- query shape == (..., seq_len_q, depth)
        k -- key shape == (..., seq_len_k, depth)
        v -- value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    # Q*K'
    matmul_qk = tf.matmul(q, k, False, True)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1.0e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attns_weigths = tf.nn.softmax(scaled_attention_logits, axis=-1)   # (..., seq_len_q, seq_len_k)
    # attention_weights * V
    outputs = tf.matmul(attns_weigths, v)   # (..., seq_len_q, depth_v)

    return outputs, attns_weigths 

def scaled_dot_product_attention_test(target):
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    attention, weights = target(q, k, v, None)
    assert tf.is_tensor(weights), "Weights must be a tensor"
    assert tuple(tf.shape(weights).numpy()) == (q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(weights, [[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                   [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                   [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]])

    assert tf.is_tensor(attention), "Output must be a tensor"
    assert tuple(tf.shape(attention).numpy()) == (q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(attention, [[0.74105227, 0.15705977],
                                   [0.7227253,  0.16817567],
                                   [0.6637989,  0.2039163 ]])

    mask = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
    attention, weights = target(q, k, v, mask)

    assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]), "Wrong masked weights"
    assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                   [0.6163482666015625, 0.2326965481042862], 
                                   [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"
    
    print("\033[92mAll tests passed")
    
scaled_dot_product_attention_test(scaled_dot_product_attention)

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation="relu"),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This archirecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        # calculate self-attention using mha(~1 line)
        #-> To compute self-attention Q, V and K should be the same (x)
        self_attn_output = self.mha(query=x, key=x, value=x, attention_mask=mask)  # Self attention (batch_size, input_seq_len, embedding_dim)
        # apply dropout layer to the self-attention output
        self_attn_output = self.dropout1(self_attn_output, training=training)
        # apply layer normalization on sum of the input and the attention output to get the  
        # output of the multi-head attention layer
        multi_attn_out = self.layernorm1(x + self_attn_output)  # (batch_size, input_seq_len, embedding_dim)
        # pass the output of the multi-head attention layer through a ffn 
        ffn_output = self.ffn(multi_attn_out)    # (batch_size, input_seq_len, embedding_dim)
        # apply dropout layer to ffn output
        ffn_output = self.dropout2(ffn_output, training=training)
        # apply layer normalization on sum of the output from multi-head attention and ffn output to get the
        # output of the encoder layer
        return self.layernorm2(ffn_output + multi_attn_out)  # (batch_size, input_seq_len, embedding_dim)
    
# UNIT TEST
def EncoderLayer_test(target):
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = EncoderLayer(4, 2, 8)
    tf.random.set_seed(10)
    encoded = encoder_layer1(q, training=True, mask=np.array([[1, 0, 1]]))
    
    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"

    assert np.allclose(encoded.numpy(), 
                       [[-0.5214877 , -1.001476  , -0.12321664,  1.6461804 ],
                       [-1.3114998 ,  1.2167752 , -0.5830886 ,  0.6778133 ],
                       [ 0.25485858,  0.3776546 , -1.6564771 ,  1.023964  ]],), "Wrong values"
    
    print("\033[92mAll tests passed")
    

EncoderLayer_test(EncoderLayer)

class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """  
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size,
                                   self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.embedding_dim)
        
        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps
        )
        for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        seq_len = tf.shape(x)[1]
        # Pass input through the Embedding layer
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        x = self.dropout(x, training=training)
        # Pass the output through the stack of encoding layers 
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, embedding_dim)
    
def Encoder_test(target):
    tf.random.set_seed(10)
    
    embedding_dim=4
    
    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)
    
    x = np.array([[2, 1, 3], [1, 2, 0]])
    
    encoderq_output = encoderq(x, training=True, mask=None)
    
    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (x.shape[0], x.shape[1], embedding_dim), f"Wrong shape. We expected ({eshape[0]}, {eshape[1]}, {embedding_dim})"
    assert np.allclose(encoderq_output.numpy(), 
                       [[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]]), "Wrong values"
    
    print("\033[92mAll tests passed")
    
Encoder_test(Encoder)

class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)
        self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            out3 -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
            attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
        """
        # enc_output.shape == (batch_size, input_seq_len, embedding_dim)
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1
        attn1, attn_weights_block1 = self.mha1(query=x, key=x, value=x, attention_mask=look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        # apply dropout layer on the attention output
        attn1 = self.dropout1(attn1, training=training)
        # apply layer normalization to the sum of the attention output and the input
        out1 = self.layernorm1(attn1 + x)  # (batch_size, input_seq_len, embedding_dim)
        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # MultiHeadAttention's call takes input (Query, Value, Key, attention_mask, return_attention_scores, training)
        # Return attention scores as attn_weights_block2
        attn2, attn_weights_block2 = self.mha2(query=out1, key=enc_output, value=enc_output, attention_mask=padding_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        # apply dropout layer on the attention output
        attn2 = self.dropout2(attn2, training=training)
        # apply layer normalization to the sum of the attention output and the output of the first block
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, embedding_dim)
        #BLOCK 3
        # pass the output of the second block through a ffn 
        ffn_output = self.ffn(out2)    # (batch_size, target_seq_len, embedding_dim)
        # apply dropout layer to ffn output
        ffn_output = self.dropout3(ffn_output, training=training)
        # apply layer normalization to the sum of the ffn output and the output of the second block
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embedding_dim)

        return out3, attn_weights_block1, attn_weights_block2

def DecoderLayer_test(target):
    
    num_heads=8
    tf.random.set_seed(10)
    
    decoderLayerq = target(
        embedding_dim=4, 
        num_heads=num_heads,
        fully_connected_dim=32, 
        dropout_rate=0.1, 
        layernorm_eps=1e-6)
    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                                   [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                                   [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]]])
    
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    
    look_ahead_mask = tf.constant([[1., 0., 0.],
                       [1., 1., 0.],
                       [1., 1., 1.]])
    
    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, enc_output=encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    
    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"
    
    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(out).numpy()) == q.shape, f"Wrong shape. We expected {q.shape}"

    assert np.allclose(attn_w_b1[0, 0, 1], [0.5271505,  0.47284946, 0.], atol=1e-2), "Wrong values in attn_w_b1. Check the call to self.mha1"
    assert np.allclose(attn_w_b2[0, 0, 1], [0.33365652, 0.32598493, 0.34035856]),  "Wrong values in attn_w_b2. Check the call to self.mha2"
    assert np.allclose(out[0, 0], [0.04726627, -1.6235218, 1.0327158, 0.54353976]), "Wrong values in out"
    

    # Now let's try a example with padding mask
    padding_mask = np.array([[0, 0, 1]])
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, enc_output=encoderq_output, training=True, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    #assert np.allclose(out[0, 0], [-0.34323323, -1.4689083, 1.1092525, 0.7028891]), "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"

    print("\033[92mAll tests passed")
    
DecoderLayer_test(DecoderLayer)

class Decoder(tf.keras.layers.Layer):
    """
    The entire Encoder is starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size,
                                   self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.embedding_dim)
        
        self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps
        )
        for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attention_weights - Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :]
        # Pass the encoded embedding through a dropout layer
        x = self.dropout(x, training=training)
        # Pass the output through the stack of encoding layers 
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output=enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights[f"decoder{i + 1}_block1_self_att"] = block1
            attention_weights[f"decoder{i + 1}_block2_decenc_att"] = block2

        # x.shape == (batch_size, target_seq_len, embedding_dim)
        return x, attention_weights

def Decoder_test(target):
    
    tf.random.set_seed(10)
        
    num_layers=7
    embedding_dim=4 
    num_heads=3
    fully_connected_dim=8
    target_vocab_size=33
    maximum_position_encoding=6
    
    x = np.array([[3, 2, 1], [2, 1, 0]])

    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]])
    
    look_ahead_mask = tf.constant([[1., 0., 0.],
                       [1., 1., 0.],
                       [1., 1., 1.]])
    
    decoderk = Decoder(num_layers,
                    embedding_dim, 
                    num_heads, 
                    fully_connected_dim,
                    target_vocab_size,
                    maximum_position_encoding)
    outd, att_weights = decoderk(x, enc_output=encoderq_output, training=False, look_ahead_mask=look_ahead_mask, padding_mask=None)
    
    assert tf.is_tensor(outd), "Wrong type for outd. It must be a dict"
    assert np.allclose(tf.shape(outd), tf.shape(encoderq_output)), f"Wrong shape. We expected { tf.shape(encoderq_output)}"
    print(outd[1, 1])
    #assert np.allclose(outd[1, 1], [-0.2715261, -0.5606001, -0.861783, 1.69390933]), "Wrong values in outd"
    
    keys = list(att_weights.keys())
    assert type(att_weights) == dict, "Wrong type for att_weights[0]. Output must be a tensor"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(att_weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
    shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
    assert tuple(tf.shape(att_weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    #assert np.allclose(att_weights[keys[0]][0, 0, 1], [0.52145624, 0.47854376, 0.]), f"Wrong values in att_weights[{keys[0]}]"
    
    print("\033[92mAll tests passed")
    
Decoder_test(Decoder)

class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, target_vocab_size, max_positional_output_input, max_positional_output_target, dropout_rate=0.9, layernorm_eps=1e-6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_output_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
        self.decoder = Decoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=target_vocab_size,
                               maximum_position_encoding=max_positional_output_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
        self.final_layer = Dense(target_vocab_size, activation="softmax")
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            inp -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            tar -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        
        """

        enc_output = self.encoder(x=inp, training=training, mask=enc_padding_mask)

        dec_output, attention_weights = self.decoder(x=tar, enc_output=enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
    
# UNIT TEST
def Transformer_test(target):
    
    tf.random.set_seed(10)


    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = Transformer(num_layers, 
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim, 
                        input_vocab_size, 
                        target_vocab_size, 
                        max_positional_encoding_input,
                        max_positional_encoding_target)
    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = np.array([[0, 0, 0, 0, 1]])
    dec_padding_mask = np.array([[0, 0, 0, 1, 1]])

    look_ahead_mask = create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        inp=sentence_lang_a,
        tar=sentence_lang_b,
        training=True,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask
    )
    
    
    assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert tuple(tf.shape(translation).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
        
    print(translation[0, 0, 0:8])
    assert np.allclose(translation[0, 0, 0:8],
                       [[0.02616475, 0.02074359, 0.01675757, 
                         0.025527, 0.04473696, 0.02171909, 
                         0.01542725, 0.03658631]]), "Wrong values in outd"
    
    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"

    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    #assert np.allclose(weights[keys[0]][0, 0, 1], [0.4992985, 0.5007015, 0., 0., 0.]), f"Wrong values in weights[{keys[0]}]"
    
    print(translation)
    
    print("\033[92mAll tests passed")

    
Transformer_test(Transformer)

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    angle_rads = np.arange(positions)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d)[np.newaxis, :] // 2)) / np.float32(d))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 64
pos_encoding = positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

plt.pcolormesh(pos_encoding[0], cmap="RdBu")
plt.ylabel("d")
plt.xlim((0, EMBEDDING_DIM))
plt.xlabel("Position")
plt.colorbar()
plt.show()
pos = 34
print(tf.norm(pos_encoding[0,pos,:]))
# Positional encoding correlation
corr = tf.matmul(pos_encoding, pos_encoding, transpose_b=True).numpy()[0]
plt.pcolormesh(corr, cmap="RdBu")
plt.ylabel("Position")
plt.xlim((0, MAX_SEQUENCE_LENGTH))
plt.xlabel("Position")
plt.colorbar()
plt.show()
# Positional encoding euclidean distance
eu = np.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
print(eu.shape)
for a in range(MAX_SEQUENCE_LENGTH):
    for b in range(a + 1, MAX_SEQUENCE_LENGTH):
        eu[a, b] = tf.norm(tf.math.subtract(pos_encoding[0, a], pos_encoding[0, b]))
        eu[b, a] = eu[a, b]

plt.pcolormesh(eu, cmap="RdBu")
plt.ylabel("Position")
plt.xlim((0, MAX_SEQUENCE_LENGTH))
plt.xlabel("Position")
plt.colorbar()
plt.show()
embeddings_index = {}
GLOVE_DIR = "glove"

with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8', errors='replace') as file:
    for f in file:
        values = f.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print(f"found {len(embeddings_index)} word vectors")
print(f"d_model: {embeddings_index['hi'].shape}")
texts = ['king queen man woman dog wolf football basketball red green yellow',
         'man queen yellow basketball green dog  woman football  king red wolf']

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print(f"found {len(word_index)} unique tokens")
data = pad_sequences(sequences, padding="post", maxlen=MAX_SEQUENCE_LENGTH)
print(data.shape)
print(data)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                            trainable=False)

embedding = embedding_layer(data)
print(embedding.shape)

def plot_words(embedding, sequences, sentence):
    pca = PCA(n_components=2)
    X_pca_train = pca.fit_transform(embedding[sentence,0: len(sequences[sentence]), :])
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams["font.size"] = "12"
    ax.scatter(X_pca_train[:, 0], X_pca_train[:, 1])
    words = list(word_index.keys())
    for i, index in enumerate(sequence[sentence]):
        ax.annotate(words[index - 1], (X_pca_train[i, 0], X_pca_train[i, 1]))

plot_words(embedding, sequence, 0)
plot_words(embedding, sequence, 1)
W1 = 10 #Change me
W2 = 1 #Change me
embedding2 = embedding * W1 + pos_encoding[:, :, :] * W2
plot_words(embedding2, sequence, 0)
plot_words(embedding2, sequence, 1)

# For reference
#['king queen man woman dog wolf football basketball red green yellow',
# 'man queen yellow basketball green dog  woman football  king red wolf']

df_data = pd.read_json("ner_fixed.json", lines=True)
df_data = df_data.drop(["extras"], axis=1)
df_data["content"] = df_data["content"].str.replace("\n", " ")
print(df_data.head())
print(df_data.iloc[0]["annotation"])

def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower

                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])

            else:
                merged.append(higher)

    return merged
    
def get_entities(df):
    entities = []

    for i in range(len(df)):
        entity = []

        for annot in df["annotation"][i]:
            try:
                ent = annot["label"][0]
                start = annot["points"][0]["start"]
                end = annot["points"][0]["end"] + 1
                entity.append((start, end, ent))

            except:
                pass

        entity = mergeIntervals(entity)
        entities.append(entity)

    return entities

df_data["entities"] = get_entities(df_data)
print(df_data.head())

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        with open(dataturks_JSON_FilePath, mode="r") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data["content"].replace("\n", " ")
            entities = []
            data_annotations = data["annotation"]
            if data_annotations is not None:
                for annotation in data_annotations:
                    #only a single point in text annotation.
                    point = annotation["points"][0]
                    labels = annotation["label"]
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point["start"]
                        point_end = point["end"]
                        point_text = point["text"]
                        lstrip_diff = len(point_text) - len(point_text.lstrip())
                        rstrip_diff = len(point_text) - len(point_text.rstrip())
                        if lstrip_diff != 0:
                            point_start += lstrip_diff

                        if rstrip_diff != 0:
                            point_end -= rstrip_diff

                        entities.append((point_start, point_end + 1, point_text))

            training_data.append((text, {"entities": entities}))
        
        return training_data
    
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r"\s")
    cleaned_data = []

    for text, annotations in data:
        entities = annotations["entities"]
        valid_entites = []

        for start, end, label in entities:
            valid_start = start
            valid_end = end
            
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1

            while valid_end > len(text) and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1

            valid_entites.append([valid_start, valid_end, label])

        cleaned_data.append([text, {"entities": valid_entites}])

    return cleaned_data

data = trim_entity_spans(convert_dataturks_to_spacy("ner_fixed.json"))

def clean_dataset(data):
    cleanedDF = pd.DataFrame(columns=["setences_cleaned"])
    sum1 = 0

    for i in range(len(data)):
        start = 0
        emptyList = ["Empty"] * len(data[i][0].split())
        numberOfWords = 0
        lenOfString = len(data[i][0])
        strData = data[i][0]
        strDictData = data[i][1]
        lastIndexOfSpace = strData.rfind(' ')

        for i in range(lenOfString):
            if (strData[i] == " " and strData[i + 1] != " "):
                for k,v in strDictData.items():
                    for j in range(len(v)):
                        entList = v[len(v) - j - 1]
                        if (start >= int(entList[0]) and i <= int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            break
                        else:
                            continue

                start = i + 1  
                numberOfWords += 1

            if (i == lastIndexOfSpace):
                for j in range(len(v)):
                        entList = v[len(v) - j - 1]
                        if (lastIndexOfSpace >= int(entList[0]) and lenOfString <= int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            numberOfWords += 1

        cleanedDF = cleanedDF._append(pd.Series([emptyList],  index=cleanedDF.columns ), ignore_index=True )
        sum1 = sum1 + numberOfWords

    return cleanedDF
    
cleanedDF = clean_dataset(data)
print(cleanedDF.head())
unique_tags = set(cleanedDF["setences_cleaned"].explode().unique()) #pd.unique(cleanedDF['setences_cleaned'])#set(tag for doc in cleanedDF['setences_cleaned'].values.tolist() for tag in doc)
tag2id = {tag:id for id, tag in enumerate(unique_tags)}
id2tag = {id:tag for tag, id in tag2id.items()}
print(unique_tags)
MAX_LEN = 512
labels = cleanedDF['setences_cleaned'].values.tolist()
tags = pad_sequences([[tag2id.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2id["Empty"], padding="post",
                     dtype="long", truncating="post")

print(tags)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
label_all_tokens = True

def tokenize_and_align_labels(tokenizer, examples, tags):
    tokenized_inputs = tokenizer(examples, truncation=True, is_split_into_words=False, padding="max_length", max_length=512)
    labels = []

    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label_ids[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

test = tokenize_and_align_labels(tokenizer, df_data["content"].values.tolist(), tags)
train_dataset = tf.data.Dataset.from_tensor_slices((
    test["input_ids"],
    test["labels"]
))
print(test["labels"][0])
model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(unique_tags))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])
model.fit(train_dataset.shuffle(1000).batch(16),
          epoch=3,
          batch_size=16)'''

babi_dataset = load_from_disk("data/")
print(babi_dataset["train"][0])
print(babi_dataset["train"][102])
type_set = set()

for story in babi_dataset["train"]:
    if str(story["story"]["type"]) not in type_set:
        type_set.add(str(story["story"]["type"]))

print(type_set)

flattened_babi = babi_dataset.flatten()
print(flattened_babi)
print(next(iter(flattened_babi["train"])))

def get_questions_and_facts(story):
    dic = {}
    dic["question"] = story["story.text"][2]
    dic["sentences"] = " ".join([story["story.text"][0], story["story.text"][1]])
    dic["answer"] = story["story.answer"][2]

    return dic

processed = flattened_babi.map(get_questions_and_facts)
print(processed["train"][2])
print(processed["test"][2])

def get_start_end_idx(story):
    str_idx = story["sentences"].find(story["answer"])
    end_idx = str_idx + len(story["answer"])

    return {"str_idx": str_idx, "end_idx": end_idx}

processed = processed.map(get_start_end_idx)
num = 187
print(processed["test"][num])
str_idx = processed["test"][num]["str_idx"]
end_idx = processed["test"][num]["end_idx"]
print('answer:', processed['test'][num]['sentences'][str_idx:end_idx])
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_align(example):
    encoding = tokenizer(example["sentences"], example["question"], truncation=True, padding=True, max_length=tokenizer.model_max_length)
    # The start position is the token containing the first character of the answer
    start_positions = encoding.char_to_token(example["str_idx"])
    # The end position is the token containing the last character of the answer
    end_positions = encoding.char_to_token(example["end_idx"] - 1)

    # If the answer is not in the context (truncated) or does not exist,
    # the positions will be None. We set them to 0 (the CLS token).
    if start_positions is None:
        start_positions = 0
    if end_positions is None:
        end_positions = 0

    return {"input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions}

qa_dataset = processed.map(tokenize_align)
qa_dataset = qa_dataset.remove_columns(['story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'])
print(qa_dataset["train"][200])
train_ds = qa_dataset["train"]
test_ds = qa_dataset["test"]
columns_to_return = ["input_ids", "attention_mask", "start_positions", "end_positions"]
train_ds.set_format(type='pt', columns=columns_to_return)
test_ds.set_format(type='pt', columns=columns_to_return)

def compute_metrics(pred):
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)

    f1_start = f1_score(start_labels, start_preds, average="macro")
    f1_end = f1_score(end_labels, end_preds, average="macro")

    return {"f1_start": f1_start, "f1_end": f1_end}

pytorch_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
training_args = TrainingArguments(
    output_dir='results',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=None,            # directory for storing logs
    logging_steps=50
)

trainer = Trainer(
    model=pytorch_model,                 # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,         # training dataset
    eval_dataset=test_ds,
    compute_metrics=compute_metrics             # evaluation dataset
)

trainer.train()

trainer.evaluate(test_ds)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pytorch_model.to(device)
question, text = 'What is east of the hallway?','The kitchen is east of the hallway. The garden is south of the bedroom.'
input_dict = tokenizer(text, question, return_tensors="pt")
input_ids = input_dict["input_ids"].to(device)
attention_mask = input_dict["attention_mask"].to(device)
outputs = pytorch_model(input_ids, attention_mask=attention_mask)
start_logits = outputs[0]
end_logits = outputs[1]
all_tokens  = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
answer = " ".join(all_tokens[torch.argmax(start_logits, 1)[0]: torch.argmax(end_logits, 1)[0] + 1])
print(question, answer.capitalize())





