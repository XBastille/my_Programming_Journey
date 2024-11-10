import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import pandas as pd
import tensorflow as tf
import pprint
import argparse
import PIL
import scipy
import os
import scipy.misc
import imageio
import pathlib
from numpy import genfromtxt
import tensorflow.keras.layers as tfl
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, ZeroPadding2D, BatchNormalization, Flatten, AveragePooling2D, MaxPool2D, GlobalMaxPooling2D, Conv2D, Activation, Dropout, Conv2DTranspose, concatenate, MaxPooling2D
from tensorflow.keras.models import Model, load_model
#from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tensorflow.keras.layers import RandomFlip, RandomRotation, Layer
from tensorflow.keras.layers import Layer
#from yad2k.models.keras_yolo import yolo_head
#from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
#from public_tests import *
from tensorflow.python.framework import ops
from matplotlib.pyplot import imread
from PIL import Image, ImageFont, ImageDraw
#from cnn_utils import *
#from test_utils import summary, comparator
np.random.seed(1)
'''plt.rcParams["figure.figsize"]=(5.0, 4.0) # set default size of plots
plt.rcParams["image.interpolation"]="nearest" 
plt.rcParams["image.cmap"]="grey" 
# GRADED FUNCTION: zero_pad
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    #(≈ 1 line)
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])
fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)
# GRADED FUNCTION: conv_single_step
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    # YOUR CODE STARTS HERE
    s=np.multiply(a_slice_prev, W)
    Z=np.sum(s)
    b=np.squeeze(b)
    Z+=b
    # YOUR CODE ENDS HERE
    return Z
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)
Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)
assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"
# GRADED FUNCTION: conv_forward
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape 
    (m, n_H_prev, n_W_prev, n_C_prev)=A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C)=W.shape
    # Retrieve information from "hparameters"
    stride=hparameters["stride"]
    pad=hparameters["pad"]
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation.
    n_H=int((n_H_prev-f+2*pad)/stride)+1
    n_W=int((n_W_prev-f+2*pad)/stride)+1
    # Initialize the output volume Z with zeros.
    Z=np.zeros((m, n_H, n_W, n_C))
    # Create A_prev_pad by padding A_prev
    A_prev_pad=zero_pad(A_prev, pad)
    for i in range(m):         # loop over the batch of training examples
        a_prev_pad=A_prev_pad[i]    # Select ith training example's padded activation
        for h in range(n_H):     # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start=stride*h
            vert_end=vert_start+f
            for w in range(n_W):    # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                    horiz_start=stride*w
                    horiz_end=horiz_start+f
                    for c in range(n_C):   # loop over channels (= #filters) of the output volume
                         # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                        a_slice_prev=a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                          # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                        weights=W[:,:,:,c]    
                        biases=b[:,:,:,c]    
                        Z[i, h, w, c]=conv_single_step(a_slice_prev, weights, biases)
    # Save information in "cache" for the backprop
    cache=(A_prev, W, b, hparameters)
    return Z, cache
np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
z_mean = np.mean(Z)
z_0_2_1 = Z[0, 2, 1]
cache_0_1_2_3 = cache_conv[0][1][2][3]
print("Z's mean =\n", z_mean)
print("Z[0,2,1] =\n", z_0_2_1)
print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
conv_forward_test_2(conv_forward)
# GRADED FUNCTION: pool_forward
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev)=A_prev.shape
    # Retrieve hyperparameters from "hparameters"
    f=hparameters["f"]
    stride=hparameters["stride"]
    # Define the dimensions of the output
    n_H=int((n_H_prev-f)/stride)+1
    n_W=int((n_W_prev-f)/stride)+1
    n_C=n_C_prev
    # Initialize output matrix A
    A=np.zeros((m, n_H, n_W, n_C))
    for i in range(m):    # loop over the training examples
        a_prev_slice=A_prev[i]
        for w in range(n_W):     # loop on the horizontal axis of the output volume
             # Find the vertical start and end of the current "slice"
            horiz_start=stride*w
            horiz_end=horiz_start+f
            for h in range(n_H):     # loop on the vertical axis of the output volume
                # Find the vertical start and end of the current "slice" 
                vert_start=stride*h
                vert_end=vert_start+f
                for c in range(n_C):      # loop over the channels of the output volume
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_slice_prev=a_prev_slice[vert_start:vert_end, horiz_start:horiz_end, c]
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    if mode=="max":
                        A[i, h, w, c]=np.max(a_slice_prev)
                    else:
                        A[i, h, w, c]=np.mean(a_slice_prev)
    cache=(A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache
# Case 1: stride of 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev, hparameters, mode = "max")
print("mode = max")
print(f"A.shape = {str(A.shape)}")
print("A[1, 1] =\n", A[1, 1])
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print(f"A.shape = {str(A.shape)}")
print("A[1, 1] =\n", A[1, 1])

pool_forward_test(pool_forward)
# Case 2: stride of 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[0] =\n", A[0])
print()

A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1] =\n", A[1])
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()
# Normalize image vectors
X_train=X_train_orig/255
X_test=X_test_orig/255
# Reshape
Y_train=Y_train_orig.T
Y_test=Y_test_orig.T
print(f"number of training examples = {str(X_train.shape[0])}")
print(f"number of test examples = {str(X_test.shape[0])}")
print(f"X_train shape: {str(X_train.shape)}")
print(f"Y_train shape: {str(Y_train.shape)}")
print(f"X_test shape: {str(X_test.shape)}")
print(f"Y_test shape: {str(Y_test.shape)}")
index=5
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
# GRADED FUNCTION: happyModel
def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    return tf.keras.Sequential([
        ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tfl.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
        ## Conv2D with 32 7x7 filters and stride of 1
        tfl.Conv2D(32, (7, 7)),
        ## BatchNormalization for axis 3
        tfl.BatchNormalization(axis=-1),
        ## ReLU
        tfl.ReLU(),
        ## Max Pooling 2D with default parameters
        tfl.MaxPool2D(),
        ## Flatten layer
        tfl.Flatten(),
        ## Dense layer with 1 unit for output & 'sigmoid' activation
        tfl.Dense(1, activation="sigmoid")
    ])
happy_model=happyModel()
for layer in summary(happy_model):
    print(layer)
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]

comparator(summary(happy_model), output)
happy_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
happy_model.summary()
happy_model.fit(X_train, Y_train, epochs=100, batch_size=16)
happy_model.evaluate(X_test, Y_test)
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print(f"y = {str(np.squeeze(Y_train_orig[:, index]))}")
X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print(f"number of training examples = {str(X_train.shape[0])}")
print(f"number of test examples = {str(X_test.shape[0])}")
print(f"X_train shape: {str(X_train.shape)}")
print(f"Y_train shape: {str(Y_train.shape)}")
print(f"X_test shape: {str(X_test.shape)}")
print(f"Y_test shape: {str(Y_test.shape)}")
def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    input_img=tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1=tfl.Conv2D(8, 4, activation="linear", padding="same", strides=1)(input_img)
    ## RELU
    A1=tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1=tfl.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding="same")(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2=tfl.Conv2D(16, 2, activation="linear", padding="same", strides=1)(P1)
    ## RELU
    A2=tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2=tfl.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding="same")(A2)
    ## FLATTEN
    F=tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs=tfl.Dense(6, activation="softmax")(F)
    return tf.keras.Model(inputs=input_img, outputs=outputs)
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
train_dataset=tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset=tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history=conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
history.history
df_loss_acc=pd.DataFrame(history.history)
df_loss=df_loss_acc(["loss", "val_loss"])
df_loss.rename(columns={"loss": "train", "val_loss": "validation"}, inplace=True)
df_acc=df_loss_acc(["accuracy", "val_accuracy"])
df_acc.rename(columns={"accuracy": "train", "val_accuracy": "validation"}, inplace=True)
df_loss.plot(title="Model Loss", figsize=(12, 8)).set(xlabel="Epoch", ylabel="Loss")
df_acc.plot(title="Model Accuracy", figsize=(12, 8)).set(xlabel="Epoch", ylabel="accuracy")
plt.show()
def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    # Retrieve Filters
    F1, F2, F3=filters
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut=X
    # First component of main path
    X=Conv2D(filters=F1, strides=(1, 1), padding="valid", kernel_size=1, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)   # Default axis
    X=Activation("relu")(X)
    ## Second component of main path 
    ## Set the padding = 'same'
    X=Conv2D(filters=F2, strides=(1, 1), padding="same", kernel_size=f, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)  
    X=Activation("relu")(X)
    ## Third component of main path 
    ## Set the padding = 'valid'
    X=Conv2D(filters=F3, strides=(1, 1), padding="valid", kernel_size=1, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)  
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation
    X=Add()([X_shortcut, X])
    X=Activation("relu")(X)
    return X
    
np.random.seed(1)
X1 = np.ones((1, 4, 4, 3)) * -1
X2 = np.ones((1, 4, 4, 3)) * 1
X3 = np.ones((1, 4, 4, 3)) * 3

X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

A3 = identity_block(X, f=2, filters=[4, 4, 3],
                   initializer=lambda seed=0:constant(value=1),
                   training=False)
print('\033[1mWith training=False\033[0m\n')
A3np = A3.numpy()
print(np.around(A3.numpy()[:,(0,-1),:,:].mean(axis = 3), 5))
resume = A3np[:,(0,-1),:,:].mean(axis = 3)
print(resume[1, 1, 0])

print('\n\033[1mWith training=True\033[0m\n')
np.random.seed(1)
A4 = identity_block(X, f=2, filters=[3, 3, 3],
                   initializer=lambda seed=0:constant(value=1),
                   training=True)
print(np.around(A4.numpy()[:,(0,-1),:,:].mean(axis = 3), 5))

identity_block_test(identity_block)

def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # Retrieve Filters
    F1, F2, F3=filters
    # Save the input value
    X_shortcut=X
    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X=Conv2D(filters=F1, strides=(s, s), padding="valid", kernel_size=1, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)   
    X=Activation("relu")(X)

    ## Second component of main path
    X=Conv2D(filters=F2, strides=(1, 1), padding="same", kernel_size=f, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)  
    X=Activation("relu")(X)

    ## Third component of main path
    X=Conv2D(filters=F3, strides=(1, 1), padding="valid", kernel_size=1, kernel_initializer=initializer(seed=0))(X)
    X=BatchNormalization(axis=3)(X, training=training)  
    ##### SHORTCUT PATH ##### 
    X_shortcut=Conv2D(filters=F3, strides=(s, s), padding="valid", kernel_size=1, kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut=BatchNormalization(axis=3)(X_shortcut, training=training)  
    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X=Add()([X, X_shortcut])
    X=Activation("relu")(X)
    return X

from outputs import convolutional_block_output1, convolutional_block_output2
np.random.seed(1)
#X = np.random.randn(3, 4, 4, 6).astype(np.float32)
X1 = np.ones((1, 4, 4, 3)) * -1
X2 = np.ones((1, 4, 4, 3)) * 1
X3 = np.ones((1, 4, 4, 3)) * 3

X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

A = convolutional_block(X, f = 2, filters = [2, 4, 6], training=False)

assert type(A) == EagerTensor, "Use only tensorflow and keras functions"
assert tuple(tf.shape(A).numpy()) == (3, 2, 2, 6), "Wrong shape."
assert np.allclose(A.numpy(), convolutional_block_output1), "Wrong values when training=False."
print(A[0])

B = convolutional_block(X, f = 2, filters = [2, 4, 6], training=True)
assert np.allclose(B.numpy(), convolutional_block_output2), "Wrong values when training=True."

print('\033[92mAll tests passed!')

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input=Input(input_shape)
    # Zero-Padding
    X=ZeroPadding2D((3, 3))(X_input)
    # Stage 1
    X=Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3)(X)
    X=Activation("relu")(X)
    X=MaxPool2D((3, 3), strides=(2, 2))(X)
    # Stage 2
    X=convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X=identity_block(X, 3, [64, 64, 256])
    X=identity_block(X, 3, [64, 64, 256])
    ## stage 3
    X=convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X=identity_block(X, 3, [128, 128, 512])
    X=identity_block(X, 3, [128, 128, 512])
    X=identity_block(X, 3, [128, 128, 512])
    ## Stage 4
    X=convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X=identity_block(X, 3, [256, 256, 1024])
    X=identity_block(X, 3, [256, 256, 1024])
    X=identity_block(X, 3, [256, 256, 1024])
    X=identity_block(X, 3, [256, 256, 1024])
    X=identity_block(X, 3, [256, 256, 1024])
    ## Stage 5
    X=convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X=identity_block(X, 3, [512, 512, 2048])
    X=identity_block(X, 3, [512, 512, 2048])
    ## AVGPOOL
    X=AveragePooling2D((2, 2))(X)
    # output layer
    X=Flatten()(X)
    X=Dense(classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0))(X)
    return Model(inputs=X_input, outputs=X)

model = ResNet50(input_shape = (64, 64, 3), classes = 6)
print(model.summary())
from outputs import ResNet50_summary
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
comparator(summary(model), ResNet50_summary)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train=X_train_orig/255
X_test=X_test_orig/255
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print(f"number of training examples = {str(X_train.shape[0])}")
print(f"number of test examples = {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=32
)
preds=model.evaluate(X_test, Y_test)
print(f"Loss={preds[0]}")
print(f"Test Accuracy={preds[1]}")
model.save("resnet50.h5")
pre_trained_model=load_model("resnet50.h5")
preds = pre_trained_model.evaluate(X_test, Y_test)
print(f"Loss = {preds[0]}")
print(f"Test Accuracy = {preds[1]}")
img_path="images/my_image.jpg"
img=image.load_img(img_path, target_size=(64, 64))
X=image.img_to_array(img)
X=np.expand_dims(X, axis=0)
X=X/255
print('Input image shape:', X.shape)
imshow(img)
prediction=pre_trained_model.predict(X)
print("Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ", prediction)
print("Class:", np.argmax(prediction))
pre_trained_model.summary()
BATCH_SIZE=32
IMG_SIZE=(160, 160)
directory="dataset/"
train_dataset=image_dataset_from_directory(
    directory,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

validation_dataset=image_dataset_from_directory(
    directory,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names=train_dataset.class_names
plt.figure(figsize=(10, 10))
for images, lables in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[lables[i]])
        plt.axis("off")
AUTOTUNE=tf.data.experimental.AUTOTUNE
train_dataset=train_dataset.prefetch(buffer_size=AUTOTUNE)
def data_augmenter():
    """
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    """
    data_augmentation=tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation

augmenter = data_augmenter()

assert(augmenter.layers[0].name.startswith('random_flip')), "First layer must be RandomFlip"
assert augmenter.layers[0].mode == 'horizontal', "RadomFlip parameter must be horizontal"
assert(augmenter.layers[1].name.startswith('random_rotation')), "Second layer must be RandomRotation"
#assert augmenter.layers[1].factor == 0.2, "Rotation factor must be 0.2"
assert len(augmenter.layers) == 2, "The model must have only 2 layers"

print('\033[92mAll tests passed!')

data_augmentation=data_augmenter()
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image=image[0]
    for i in range(9):
        plt.subplot(3, 3, i+1)
        augmented_image=data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0]/255)
        plt.axis()
preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input
IMG_SHAPE=IMG_SIZE+(3,)
base_model=tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=True,
    weights="imagenet"
)
base_model.summary()
nb_layers=len(base_model.layers)
print(base_model.layers[nb_layers-2].name)
print(base_model.layers[nb_layers-1].name)
image_batch, label_batch=next(iter(train_dataset))
feature_batch=base_model(image_batch)
print(feature_batch.shape)
#Shows the different label probabilities in one tensor 
print(label_batch)
base_model.trainable=False
image_var=tf.Variable(preprocess_input(image_batch))
pred=base_model(image_var)
print(tf.keras.applications.mobilenet_v2.decode_predictions(pred.numpy(), top=2))
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    """ Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    """
    input_shape=image_shape+(3,)
    base_model=tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,   # <== Important!!!!
        weights="imagenet"    # From imageNet
    )
    # freeze the base model by making it non trainable
    base_model.trainable=False   
    # create the input layer (Same as the imageNetv2 input size)
    inputs=tf.keras.Input(shape=input_shape)
    # apply data augmentation to the inputs
    X=data_augmentation(inputs)
    # data preprocessing using the same weights the model was trained on
    X=preprocess_input(X)
    # set training to False to avoid keeping track of statistics in the batch norm layer
    X=base_model(X, training=False)
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    X=tfl.GlobalAveragePooling2D()(X)
    # include dropout with probability of 0.2 to avoid overfitting
    X=tfl.Dropout(0.2)(X)
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    prediction_layer=tfl.Dense(1)
    outputs=prediction_layer(X)
    return tf.keras.Model(inputs, outputs)

model2=alpaca_model(IMG_SIZE, data_augmentation)

from test_utils import summary, comparator

alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
                    ['Sequential', (None, 160, 160, 3), 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['Functional', (None, 5, 5, 1280), 2257984],
                    ['GlobalAveragePooling2D', (None, 1280), 0],
                    ['Dropout', (None, 1280), 0, 0.2],
                    ['Dense', (None, 1), 1281, 'linear']] #linear is the default activation

comparator(summary(model2), alpaca_summary)
for layer in summary(model2):
    print(layer)
base_learning_rate=0.001
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
initial_epochs=5
history=model2.fit(
    train_dataset, 
    validation_data=validation_dataset, 
    epochs=initial_epochs
)
acc=[0.]+history.history["accuracy"]
val_acc=[0.]+history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")
plt.subplot(2, 1, 2)
plt.plot(loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation loss")
plt.xlabel("epoch")
plt.show()
print(class_names)
base_model=model2
base_model.trainable=True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards
fine_tune_at=120
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable=None
# Define a BinaryCrossentropy loss function. Use from_logits=True
loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer=tf.keras.optimizers.Adam(learning_rate=0.1*base_learning_rate)
# Use accuracy as evaluation metric
metrics=["accuracy"]
model2.compile(
    loss=loss_function,
    optimizer=optimizer,
    metrics=metrics
)
assert type(loss_function) == tf.keras.losses.BinaryCrossentropy, "Not the correct layer"
assert loss_function.from_logits, "Use from_logits=True"
assert type(optimizer) == tf.keras.optimizers.Adam, "This is not an Adam optimizer"
assert optimizer.learning_rate == base_learning_rate / 10, "Wrong learning rate"
assert metrics[0] == 'accuracy', "Wrong metric"

print('\033[92mAll tests passed!')

fine_tune_epochs=5
total_epochs=initial_epochs+fine_tune_epochs
history_fine=model2.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset
)
acc+=history_fine.history["accuracy"]
val_acc+=history_fine.history["val_accuracy"]
loss+=history_fine.history["loss"]
val_loss+=history_fine.history["val_loss"]
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.ylim([0, 1])
plt.plot([initial_epochs-1, initial_epochs-1],
plt.ylim(), label="Start Fine Tuning")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(2, 1, 2)
plt.plot(loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.legend(loc="upper right")
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1, initial_epochs-1],
plt.ylim(), label="Start Fine Tuning")
plt.title("Training and Validation loss")
plt.xlabel("epoch")
plt.show()'''

'''def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    # Step 1: Compute box scores
    box_scores=box_confidence*box_class_probs
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes=tf.argmax(box_scores, axis=-1)
    box_class_scores=tf.reduce_max(box_scores, axis=-1)
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask=box_class_scores>=threshold
     # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores=tf.boolean_mask(box_class_scores, filtering_mask)
    boxes=tf.boolean_mask(boxes, filtering_mask)
    classes=tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)
print(f"scores[2] = {scores[2].numpy()}")
print(f"boxes[2] = {boxes[2].numpy()}")
print(f"classes[2] = {classes[2].numpy()}")
print(f"scores.shape = {scores.shape}")
print(f"boxes.shape = {boxes.shape}")
print(f"classes.shape = {classes.shape}")

assert type(scores) == EagerTensor, "Use tensorflow functions"
assert type(boxes) == EagerTensor, "Use tensorflow functions"
assert type(classes) == EagerTensor, "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!")

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """
    (box1_x1, box1_y1, box1_x2, box1_y2)=box1
    (box2_x1, box2_y1, box2_x2, box2_y2)=box2
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1=max(box1_x1, box2_x1)
    yi1=max(box1_y1, box2_y1)
    xi2=min(box1_x2, box2_x2)
    yi2=min(box1_y2, box2_y2)
    inter_height=max(0, xi2-xi1)
    inter_width=max(0, yi2-yi1)
    inter_area=inter_width*inter_height
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area=(box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area=(box2_x2-box2_x1)*(box2_y2-box2_y1)
    union_area=box1_area+box2_area-inter_area
    return inter_area/union_area
    
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

## Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

## Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

## Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"

print("\033[92m All tests passed!")

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    max_boxes_tensor=tf.Variable(max_boxes, dtype="int32")      # tensor to be used in tf.image.non_max_suppression()
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices=tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    #  Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores=tf.gather(scores, nms_indices)
    boxes=tf.gather(boxes, nms_indices)
    classes=tf.gather(classes, nms_indices)
    return scores, boxes, classes

tf.random.set_seed(10)
scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

assert type(scores) == EagerTensor, "Use tensoflow functions"
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert type(scores) == EagerTensor, "Use tensoflow functions"
assert type(boxes) == EagerTensor, "Use tensoflow functions"
assert type(classes) == EagerTensor, "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [ 6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

print("\033[92m All tests passed!")

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins=box_xy-(box_wh/2.)
    box_maxes=box_xy+(box_wh/2.)
    return tf.keras.backend.concatenate([
        box_mins[..., 1:2], #y_min
        box_mins[..., 0:1], #x_min
        box_maxes[..., 1:2], #y_max
        box_maxes[..., 0:1] #x_max
    ])

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # Retrieve outputs of the YOLO model
    box_xy, box_wh, box_confidence, box_class_probs=yolo_outputs
    # Convert boxes to be ready for filtering functions
    boxes=yolo_boxes_to_corners(box_xy, box_wh)
    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold        
    scores, boxes, classes=yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
     # Scale boxes back to original image shape.
    boxes=scale_boxes(boxes, image_shape)
    # Use one of the functions you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold
    return yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

tf.random.set_seed(10)
yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print(f"scores.shape = {str(scores.numpy().shape)}")
print(f"boxes.shape = {str(boxes.numpy().shape)}")
print(f"classes.shape = {str(classes.numpy().shape)}")

assert type(scores) == EagerTensor, "Use tensoflow functions"
assert type(boxes) == EagerTensor, "Use tensoflow functions"
assert type(classes) == EagerTensor, "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"

print("\033[92m All tests passed!")

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608)   # Same as yolo_model input layer size
yolo_model = load_model("model_data/", compile=False)
yolo_model.summary()

def predict(image_file):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """
    image, image_data = preprocess_image(f"images/{image_file}", model_image_size=(608, 608))
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    out_scores, out_classes, out_boxes=yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)
    print(f"found {len(out_boxes)} boxes for images/{image_file}")
    # Generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))
    # Draw bounding boxes on the image file
    #draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=100)
    output_image=Image.open(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes
    
out_scores, out_boxes, out_classes = predict("test.jpg")
path=""
image_path=os.path.join(path, "./data/CameraRGB/")
mask_path=os.path.join(path, "./data/CameraMask/")
image_list=os.listdir(image_path)
mask_list=os.listdir(mask_path)
image_list=[image_path+i for i in image_list]
mask_list=[mask_path+i for i in mask_list]
N=2
img=imageio.imread(image_list[N])
mask=imageio.imread(mask_list[N])
fig, arr=plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img)
arr[0].set_title("Image")
arr[1].imshow(mask[:, :, 0])
arr[1].set_title("Segmentation")
image_list_ds=tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds=tf.data.Dataset.list_files(mask_list, shuffle=False)
for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)
image_filenames=tf.constant(image_list)
masks_filenames=tf.constant(mask_list)
dataset=tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
for image, mask in dataset.take(1):
    print(image)
    print(mask)

def process_path(image_path, mask_path):
    img=tf.io.read_file(image_path)
    img=tf.image.decode_png(img, channels=3)
    img=tf.image.convert_image_dtype(img, tf.float32)

    mask=tf.io.read_file(mask_path)
    mask=tf.image.decode_png(mask, channels=3)
    mask=tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image=tf.image.resize(image, (96, 128), method="nearest")
    input_mask=tf.image.resize(mask, (96, 128), method="nearest")
    return input_image, input_mask

image_ds=dataset.map(process_path)
processed_image_ds=image_ds.map(preprocess)

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    conv=Conv2D(n_filters,  # Number of filters
        3,  # Kernel size 
        activation="relu",
        padding="same",
        kernel_initializer="he_normal")(inputs)

    conv=Conv2D(n_filters,   # Number of filters
        3, # Kernel size 
        activation="relu",
        padding="same",
        kernel_initializer="he_normal")(conv)
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob>0:
        conv=Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    next_layer = MaxPooling2D(2,strides=2)(conv) if max_pooling else conv
    skip_connection=conv

    return next_layer, skip_connection 

input_size=(96, 128, 3)
n_filters = 32
inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=inputs, outputs=cblock1)

output1 = [['InputLayer', [(None, 96, 128, 3)], 0],
            ['Conv2D', (None, 96, 128, 32), 896, 'same', 'relu', 'HeNormal'],
            ['Conv2D', (None, 96, 128, 32), 9248, 'same', 'relu', 'HeNormal'],
            ['MaxPooling2D', (None, 48, 64, 32), 0, (2, 2)]]

print('Block 1:')
for layer in summary(model1):
    print(layer)

#comparator(summary(model1), output1)

inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
model2 = tf.keras.Model(inputs=inputs, outputs=cblock1)

output2 = [['InputLayer', [(None, 96, 128, 3)], 0],
            ['Conv2D', (None, 96, 128, 1024), 28672, 'same', 'relu', 'HeNormal'],
            ['Conv2D', (None, 96, 128, 1024), 9438208, 'same', 'relu', 'HeNormal'],
            ['Dropout', (None, 96, 128, 1024), 0, 0.1],
            ['MaxPooling2D', (None, 48, 64, 1024), 0, (2, 2)]]
           
print('\nBlock 2:')   
for layer in summary(model2):
    print(layer)
    
#comparator(summary(model2), output2)

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    up = Conv2DTranspose(
        n_filters,    # number of filters
        3,       # Kernel size
        strides=2,
        padding="same"
    )(expansive_input)
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(
        n_filters, # Number of filters
        3,   # Kernel size  
        padding="same",
        activation="relu",
        kernel_initializer="he_normal"
    )(merge)
    conv = Conv2D(
        n_filters, # Number of filters
        3,   # Kernel size  
        padding="same",
        activation="relu",
        kernel_initializer="he_normal"
    )(conv)

    return conv

input_size1=(12, 16, 256)
input_size2 = (24, 32, 128)
n_filters = 32
expansive_inputs = Input(input_size1)
contractive_inputs =  Input(input_size2)
cblock1 = upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)

output1 = [['InputLayer', [(None, 12, 16, 256)], 0],
            ['Conv2DTranspose', (None, 24, 32, 32), 73760],
            ['InputLayer', [(None, 24, 32, 128)], 0],
            ['Concatenate', (None, 24, 32, 160), 0],
            ['Conv2D', (None, 24, 32, 32), 46112, 'same', 'relu', 'HeNormal'],
            ['Conv2D', (None, 24, 32, 32), 9248, 'same', 'relu', 'HeNormal']]

print('Block 1:')
for layer in summary(model1):
    print(layer)

#comparator(summary(model1), output1)

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs=Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    cblock1=conv_block(inputs, n_filters*1)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2=conv_block(cblock1[0], n_filters=n_filters*2)
    cblock3=conv_block(cblock2[0], n_filters=n_filters*4)
    cblock4=conv_block(cblock3[0], n_filters=n_filters*8, dropout_prob=0.3)   # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5=conv_block(cblock4[0], n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False)  
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ublock6=upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    # At each step, use half the number of filters of the previous block 
    ublock7=upsampling_block(ublock6, cblock3[1], n_filters*4)
    ublock8=upsampling_block(ublock7, cblock2[1], n_filters*2)
    ublock9=upsampling_block(ublock8, cblock1[1], n_filters*1)
    conv9=Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal"
    )(ublock9)
    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv10=Conv2D(n_classes, 1, padding="same")(conv9)

    return tf.keras.Model(inputs=inputs, outputs=conv10)

import outputs
img_height = 96
img_width = 128
num_channels = 3

unet = unet_model((img_height, img_width, num_channels))
#comparator(summary(unet), outputs.unet_model_output)
unet.summary()
unet.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title=["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

for image, mask in image_ds.take(1):
    sample_image, sample_mask=image, mask
    print(mask.shape)
display([sample_image, sample_mask])

for image, mask in processed_image_ds.take(1):
    sample_image, sample_mask=image, mask
    print(mask.shape)
display([sample_image, sample_mask])
EPOCHS=40
VAL_SUBSPLITS=4
BATCH_SIZE=32
BUFFER_SIZE=500
processed_image_ds.batch(BATCH_SIZE)
train_dataset=processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
#model_history=unet.fit(train_dataset, epochs=EPOCHS)
#unet.save('my_unet_model.h5')
unet= load_model('my_unet_model.h5')

def create_mask(pred_mask):
    pred_mask=tf.argmax(pred_mask, axis=-1)
    pred_mask=pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask=unet.predict(image)
            display([image, mask, create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(unet.predict(sample_image[tf.newaxis, ...]))])


show_predictions(train_dataset, 6)
K.set_image_data_format("channels_last")
from tensorflow.keras.models import model_from_json
loaded_model_json = pathlib.Path("keras-facenet-h5/model.json").read_text()
model=model_from_json(loaded_model_json)
model.load_weights("keras-facenet-h5/model.h5")
print(model.inputs)
print(model.outputs)

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative=y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist=tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist=tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    basic_loss=tf.maximum(tf.add(tf.subtract(pos_dist, neg_dist), alpha), 0)
    return tf.reduce_sum(basic_loss)

tf.random.set_seed(1)
y_true = (None, None, None) # It is not used
y_pred = (tf.keras.backend.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=1, stddev=1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=3, stddev=4, seed = 1))
loss = triplet_loss(y_true, y_pred)

assert type(loss) == tf.python.framework.ops.EagerTensor, "Use tensorflow functions"
print("loss = " + str(loss))

y_pred_perfect = ([1., 1.], [1., 1.], [1., 1.,])
loss = triplet_loss(y_true, y_pred_perfect, 5)
assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
y_pred_perfect = ([1., 1.],[1., 1.], [0., 0.,])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
y_pred_perfect = ([1., 1.],[0., 0.], [1., 1.,])
loss = triplet_loss(y_true, y_pred_perfect, 0)
assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
y_pred_perfect = ([0., 0.],[0., 0.], [0., 0.,])
loss = triplet_loss(y_true, y_pred_perfect, -2)
assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"
y_pred_perfect = ([[1., 1.], [2., 0.]], [[0., 3.], [1., 1.]], [[1., 0.], [0., 1.,]])
loss = triplet_loss(y_true, y_pred_perfect, 1)
if (loss == 4.):
    raise Exception('Perhaps you are not using axis=-1 in reduce_sum?')
assert loss == 5, "Wrong value. Check your implementation"

FRmodel=model

def img_to_encoding(image_path, model):
    img=tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img=np.around(np.array(img)/255.0, decimals=12)
    x_train=np.expand_dims(img, axis=0)
    embedding=model.predict_on_batch(x_train)
    return embedding/np.linalg.norm(embedding, ord=2)

database = {"danielle": img_to_encoding("images/danielle.png", FRmodel)}
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
danielle=tf.keras.preprocessing.image.load_img("images/danielle.png", target_size=(160, 160))
kian=tf.keras.preprocessing.image.load_img("images/kian.jpg", target_size=(160, 160))
print(np.around(np.array(kian)/255.0, decimals=12).shape)
print(np.around(np.array(danielle)/255.0, decimals=12).shape)

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    #Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. 
    encoding=img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image
    dist=np.linalg.norm(encoding-database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open 
    if dist<0.7:
        print(f"it's {str(identity)}, welcome in!")
        door_open=True
    else:
        print(f"it's not {str(identity)}, please go away")
        door_open=False

    return dist, door_open

verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras
    
    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding=img_to_encoding(image_path, model)
    ## Step 2: Find the closest encoding ##
    # Initialize "min_dist" to a large value, say 100
    min_dist=100
    #Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist=np.linalg.norm(encoding-db_enc)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist<min_dist:
            min_dist=dist
            identity=name
    print("it's not in the database") if min_dist>0.7 else print(f"it's {str(identity)}, the distance is {str(min_dist)}")
    return min_dist, identity

# BEGIN UNIT TEST
# Test 1 with Younes pictures 
who_is_it("images/camera_0.jpg", database, FRmodel)

# Test 2 with Younes pictures 
test1 = who_is_it("images/camera_0.jpg", database, FRmodel)
assert np.isclose(test1[0], 0.60197866)
assert test1[1] == 'younes'

# Test 3 with Younes pictures 
test2 = who_is_it("images/younes.jpg", database, FRmodel)
assert np.isclose(test2[0], 0.0)
assert test2[1] == 'younes'
# END UNIT TEST'''
tf.random.set_seed(272)
img_size=400
pp=pprint.PrettyPrinter(indent=4)
vgg=tf.keras.applications.VGG19(
    include_top=False,
    input_shape=(img_size, img_size, 3),
    weights="pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
vgg.trainable=False
pp.pprint(vgg)
content_image=Image.open("images/louvre.jpg")
print("The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.")

def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C=content_output[-1]
    a_G=generated_output[-1]
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C=a_G.get_shape().as_list()
    # Reshape a_C and a_G
    a_C_unrolled=tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled=tf.reshape(a_G, shape=[m, -1, n_C])
    # compute the cost with tensorflow
    return tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))/(4.0*n_H*n_C*n_W)

tf.random.set_seed(1)
a_C = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
a_G = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
J_content = compute_content_cost(a_C, a_G)
J_content_0 = compute_content_cost(a_C, a_C)
assert type(J_content) == EagerTensor, "Use the tensorflow function"
assert np.isclose(J_content_0, 0.0), "Wrong value. compute_content_cost(A, A) must be 0"
assert np.isclose(J_content, 7.0568767), f"Wrong value. Expected {7.0568767},  current{J_content}"

print("J_content = " + str(J_content))

# Test that it works with symbolic tensors
ll = tf.keras.layers.Dense(8, activation='relu', input_shape=(1, 4, 4, 3))
model_tmp = tf.keras.models.Sequential()
model_tmp.add(ll)
try:
    compute_content_cost(ll.output, ll.output)
    print("\033[92mAll tests passed")
except Exception as inst:
    print("\n\033[91mDon't use the numpy API inside compute_content_cost\n")
    print(inst)

example = Image.open("images/monet_800600.jpg")

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """  
    return tf.matmul(A, tf.transpose(A))


tf.random.set_seed(1)
A = tf.random.normal([3, 2 * 1], mean=1, stddev=4)
GA = gram_matrix(A)

assert type(GA) == EagerTensor, "Use the tensorflow function"
assert GA.shape == (3, 3), "Wrong shape. Check the order of the matmul parameters"
assert np.allclose(GA[0,:], [63.1888, -26.721275, -7.7320204]), "Wrong values."

print("GA = \n" + str(GA))

print("\033[92mAll tests passed")

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C=a_G.get_shape().as_list()
    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S=tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G=tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    # Computing gram_matrices for both images S and G 
    GS=gram_matrix(a_S)
    GG=gram_matrix(a_G)
    # Computing the loss
    return tf.reduce_sum(tf.square(GS-GG))/(4.0*((n_C*n_H*n_W)**2))

tf.random.set_seed(1)
a_S = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
J_style_layer_GG = compute_layer_style_cost(a_G, a_G)
J_style_layer_SG = compute_layer_style_cost(a_S, a_G)


assert type(J_style_layer_GG) == EagerTensor, "Use the tensorflow functions"
assert np.isclose(J_style_layer_GG, 0.0), "Wrong value. compute_layer_style_cost(A, A) must be 0"
assert J_style_layer_SG > 0, "Wrong value. compute_layer_style_cost(A, B) must be greater than 0 if A != B"
assert np.isclose(J_style_layer_SG, 14.017805), "Wrong value."

print("J_style_layer = " + str(J_style_layer_SG))

for layers in vgg.layers:
    print(layers.name)

print(vgg.get_layer("block5_conv4").output)
STYLE_LAYERS=[
    ("block1_conv1", 0.2),
    ("block2_conv1", 0.2),
    ("block3_conv1", 0.2),
    ("block4_conv1", 0.2),
    ("block5_conv1", 0.2),
]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style=0
    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S=style_image_output[:-1]
    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G=generated_image_output[:-1]
    for i , weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer=compute_layer_style_cost(a_S[i], a_G[i])
        # Add weight * J_style_layer of this layer to overall style cost
        J_style+=weight[1]*J_style_layer
    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    return alpha*J_content+beta*J_style

J_content = 0.2    
J_style = 0.8
J = total_cost(J_content, J_style)

assert type(J) == EagerTensor, "Do not remove the @tf.function() modifier from the function"
assert J == 34, "Wrong value. Try inverting the order of alpha and beta in the J calculation"
assert np.isclose(total_cost(0.3, 0.5, 3, 8), 4.9), "Wrong value. Use the alpha and beta parameters"

np.random.seed(1)
print("J = " + str(total_cost(np.random.uniform(0, 1), np.random.uniform(0, 1))))

print("\033[92mAll tests passed")

content_image=np.array(Image.open("images/louvre_small.jpg").resize((img_size, img_size)))
content_image=tf.constant(np.reshape(content_image, ((1,)+content_image.shape)))
print(content_image.shape)
imshow(content_image[0])
plt.show()

style_image=np.array(Image.open("images/monet.jpg").resize((img_size, img_size)))
style_image=tf.constant(np.reshape(style_image, ((1,)+style_image.shape)))
print(style_image.shape)
imshow(style_image[0])
plt.show()

generated_image=tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise=tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image=tf.add(generated_image, noise)
generated_image=tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs=[vgg.get_layer(layer[0]).output for layer in layer_names]
    return tf.keras.Model([vgg.input], outputs)

content_layer=[("block5_conv4", 1)]
vgg_model_outputs=get_layer_outputs(vgg, STYLE_LAYERS+content_layer)
content_targets=vgg_model_outputs(content_image)  # Content encoder
style_targets=vgg_model_outputs(style_image)    # style encoder
# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content=tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C=vgg_model_outputs(preprocessed_content)
# Assign the input of the model to be the "style" image 
preprocessed_content=tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S=vgg_model_outputs(preprocessed_content)

def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor*=255
    tensor=np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0]==1
        tensor=tensor[0]
    return Image.fromarray(tensor)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G=vgg_model_outputs(generated_image)
        # Compute the style cost
        J_style=compute_style_cost(a_S, a_G)
        # Compute the content cost
        J_content=compute_content_cost(a_C, a_G)
        # Compute the total cost
        J=total_cost(J_content, J_style)
    grad=tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    # For grading purposes
    return J

tf.config.run_functions_eagerly(True)

# You always must run the last cell before this one. You will get an error if not.
generated_image = tf.Variable(generated_image)


J1 = train_step(generated_image)
print(J1)
assert type(J1) == EagerTensor, f"Wrong type {type(J1)} != {EagerTensor}"
assert np.isclose(J1, 25629.055, rtol=0.05), f"Unexpected cost for epoch 0: {J1} != {25629.055}"

J2 = train_step(generated_image)
print(J2)
assert np.isclose(J2, 17812.627, rtol=0.05), f"Unexpected cost for epoch 1: {J2} != {17735.512}"

print("\033[92mAll tests passed")
# Show the generated image at some epochs
# Uncoment to reset the style transfer process. You will need to compile the train_step function again 
epochs=2501
for i in range(epochs):
    train_step(generated_image)
    if i%250==0:
        print(f"Epoch {i} ")
        image=tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/image_{i}.jpg")
        plt.show()
# Show the 3 images in a row
fig=plt.figure(figsize=(16, 4))
ax=fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text("Content image")
ax=fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text("Style image")
ax=fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text("Generated image")



