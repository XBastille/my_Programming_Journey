import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBModel
import tensorflow as tf
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from keras import layers, losses, activations, Sequential, Input, optimizers, regularizers
from lab_utils_common import dlc, sigmoid
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
from lab_utils_softmax import plt_softmax
from lab_utils_multiclass_TF import *
from public_tests_a1 import * 
from assigment_utils import *
from public_tests import *
from utils import *
#import utils
#from autils import *
from lab_utils_relu import *
import warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action="ignore", category=UserWarning)
'''X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)
fig, ax=plt.subplots(1, 1)
ax.scatter(X_train, Y_train, c="r", marker="x", label="Data points")
ax.legend(fontsize="xx-large")
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()
linear_layer=layers.Dense(units=1, activation="linear")
linear_layer.get_weights()
a1=linear_layer(X_train[0].reshape(1,1))
print(a1)
w, b=linear_layer.get_weights()
print(f"w = {w}, b={b}")
set_w = np.array([[200]])
set_b = np.array([100])
# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b
plt_linear(X_train, Y_train, prediction_tf, prediction_np)
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
pos = Y_train == 1
neg = Y_train == 0
fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)
ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()
model=Sequential(
    [
        layers.Dense(1, input_dim=1, activation="sigmoid", name="L1")
    ]
)
print(model.summary())
logistic_layer=model.get_layer("L1")
w, b=logistic_layer.get_weights()
print(w, b)
print(w.shape, b.shape)
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())
al=model.predict(X_train)
print(a1)
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(alog)
plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)
X,Y = load_coffee_data();
print(X.shape, Y.shape)
plt_roast(X, Y)
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_1=layers.Normalization(axis=-1)
norm_1.adapt(X)  # learns mean, variance
Xn=norm_1(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
Xt=np.tile(Xn, (1000, 1))
Yt=np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)
tf.random.set_seed(1234)  #applied to achieve consistent results
model=Sequential(
    [
        Input(shape=(2, )),
        layers.Dense(3, activation="sigmoid", name="layer1"),    
        layers.Dense(1, activation="sigmoid", name="layer2"),    
    ]
)
model.summary()
L1_num_params=2*3+3    # W1 parameters  + b1 parameters
L2_num_params=3*1+1    # W2 parameters + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )
W1, b1= model.get_layer("layer1").get_weights()
W2, b2= model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.01)
)
model.fit(
    Xt, Yt, 
    epochs=10
)
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1, b1])
model.get_layer("layer2").set_weights([W2, b2])
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn=norm_1(X_test)
prediction=model.predict(X_testn)
print("predictions = \n", prediction)
yhat = np.zeros_like(prediction)
for i in range(len(prediction)):
    yhat[i]=1 if prediction[i]>=0.5 else 0
print(f"decisions = \n{yhat}")
yhat=(prediction>=0.5).astype(int)
print(f"decisions = \n{yhat}")
plt_layer(X,Y.reshape(-1,),W1,b1,norm_1)
plt_output_unit(W2,b2)
netf= lambda x : model.predict(norm_1(x))
plt_network(X,Y,netf)
X,Y = load_coffee_data();
print(X.shape, Y.shape)
plt_roast(X,Y)
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l=layers.Normalization(axis=-1)
norm_l.adapt(X)   # learns mean, variance
Xn=norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
# Define the activation function
g=sigmoid
def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    unit=W.shape[1]
    a_out=np.zeros(unit)
    for j in range(unit):
        w=W[:, j]
        z=np.dot(a_in, w)+b[j]
        a_out[j]=g(z)
    return a_out
def my_sequential(x, W1, b1, W2, b2):
    a1=my_dense(x, W1, b1)
    return my_dense(a1, W2, b2)
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )
def my_predict(X, W1, b1, W2, b2):
    m=X.shape[0]
    p=np.zeros((m, 1))
    for i in range(m):
        p[i, 0]=my_sequential(X[i], W1, b1, W2, b2)
    return p
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
yhat=(predictions>=0.5).astype(int)
print(f"decisions = \n{yhat}")
netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X,Y,netf)
# load dataset
X, y = load_data()
print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
print(f'The shape of X is: {str(X.shape)}')
print(f'The shape of y is: {str(y.shape)}')
warnings.simplefilter(action="ignore", category=FutureWarning)
m, n=X.shape
fig, axes=plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)
for i, ax in enumerate(axes.flat):
    random_index=np.random.randint(m)
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped=X[random_index].reshape((20, 20)).T
    # Display the image
    ax.imshow(X_random_reshaped, cmap="grey")
    # Display the label above the image
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
plt.show()
# UNQ_C1
# GRADED CELL: Sequential model
model=Sequential([
     Input(shape=(400,)),
    layers.Dense(
        units=25, activation="sigmoid"
    ),     
    layers.Dense(
        units=15, activation="sigmoid"
    ),     
    layers.Dense(
        units=1, activation="sigmoid"
    )  
]   , name="my_model"
)
model.summary()
L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )
[layer1, layer2, layer3]=model.layers
W1,b1=layer1.get_weights()
W2,b2=layer2.get_weights()
W3,b3=layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
print(model.layers[2].weights)
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(0.001)
)
model.fit(
    X, y,
    epochs=20
)
prediction=model.predict(X[0].reshape(1, 400))  # a zero
print(f" predicting a zero: {prediction}")
prediction=model.predict(X[500].reshape(1, 400))  # a one
print(f" predicting a one: {prediction}")
yhat=(prediction>=0.5).astype(int)
print(f"prediction after threshold: {yhat}")
m, n = X.shape
fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])    #[left, bottom, right, top]
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    # Predict using the Neural Network
    prediction=model.predict(X[random_index].reshape(1, 400))
    yhat=(prediction>=0.5).astype(int)
    ax.set_title(f"{y[random_index, 0]},{yhat[0,0]}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()
# UNQ_C2
# GRADED FUNCTION: my_dense

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units=W.shape[1]
    a_out=np.zeros(units)
    for j in range(units):
        w=W[:, j]
        z=np.dot(w, a_in)+b[j]
        a_out[j]=g(z)
    return a_out
# Quick Check
x_tst = 0.1*np.arange(1,3,1).reshape(2,)  # (1 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
print(A_tst)
# UNIT TESTS
from public_tests import *
test_c2(my_dense)
def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1=my_dense(x, W1, b1, sigmoid)
    a2=my_dense(a1, W2, b2, sigmoid)
    return my_dense(a2, W3, b3, sigmoid)
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
yhat=(prediction>=0.5).astype(int)
print( "yhat = ", yhat, " label= ", y[0,0])
prediction = my_sequential(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
yhat=(prediction>=0.5).astype(int)
print( "yhat = ", yhat, " label= ", y[500,0])
m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network implemented in Numpy
    my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    my_yhat = int(my_prediction >= 0.5)
    # Predict using the Neural Network implemented in Tensorflow
    tf_prediction = model.predict(X[random_index].reshape(1,400))
    tf_yhat = int(tf_prediction >= 0.5)
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
    ax.set_axis_off() 
fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
plt.show()
x = X[0].reshape(-1,1)         # column vector (400,1)
z1 = np.matmul(x.T,W1) + b1    # (1,400)(400,25) = (1,25)
a1 = sigmoid(z1)
print(a1.shape)
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
    return g(A_in@W+b)
X_tst = 0.1*np.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(1,3) # (1, 3 features)
A_tst = my_dense_v(X_tst, W_tst, b_tst, sigmoid)
print(A_tst)
# UNIT TESTS
from public_tests import *
test_c3(my_dense_v)
def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    return my_dense_v(A2, W3, b3, sigmoid)
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
print(Prediction.shape)
Yhat=(Prediction>=0.5).astype(int)
print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])
m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
   
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
    ax.set_axis_off() 
fig.suptitle("Label, Yhat", fontsize=16)
plt.show()
fig=plt.figure(figsize=(1,1))
error=np.where(Yhat!=y)
ind=error[0][0]
X_reshaped=X[ind].reshape(20, 20).T
plt.imshow(X_reshaped, cmap="grey")
plt.title(f"{y[ind ,0]}, {Yhat[ind, 0]}")
plt.axis('off')
plt.show()
plt_act_trio()
_ = plt_relu_ex()
def my_softmax(z):
    ez=np.exp(z)       #element-wise exponenial
    return ez/np.sum(ez)
plt.close("all")
plt_softmax(my_softmax)
print("hello")
# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train=make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
model=Sequential(
    [
    Dense(25, activation="sigmoid"),
    Dense(15, activation="sigmoid"),
    Dense(4, activation="softmax")   # < softmax activation here
]
)
model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.Adam(0.001),
)
model.fit(
    X_train, y_train,
    epochs=10
)
p_nonpreferred=model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))
preferred_model=Sequential(
    [
    Dense(25, activation="sigmoid"),
    Dense(15, activation="sigmoid"),
    Dense(4, activation="linear")  #<-- Note
]
)
preferred_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),   #<-- Note
    optimizer=optimizers.Adam(0.001),
)
preferred_model.fit(
    X_train, y_train,
    epochs=10
)
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))
sm_preferred=tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))
for i in range(5):
    print(f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")
np.set_printoptions(precision=2)
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train=make_blobs(centers=centers, cluster_std=std, n_samples=m, random_state=30)
plt_mc(X_train,y_train,classes, centers, std=std)
# show classes in data set
print(f"unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}")
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")
tf.random.set_seed(1234)   # applied to achieve consistent results
model=Sequential(
    [
        Dense(2, activation="relu", name="L1"),
        Dense(4, activation="linear", name="L2")
    ]
)
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(0.01)
)
model.fit(
    X_train, y_train,
    epochs=200
)
plt_cat_mc(X_train, y_train, model, classes)
# gather the trained parameters from the first layer
l1=model.get_layer("L1")
W1, b1= l1.get_weights()
# plot the function of the first layer
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)
# gather the trained parameters from the output layer
l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
# create the 'new features', the training examples after L1 transformation
Xl2=np.maximum(0, np.dot(X_train, W1)+b1)
plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
                        x0_rng = (-0.25,np.amax(Xl2[:,0])), x1_rng = (-0.25,np.amax(Xl2[:,1])))
from public_tests import * 
np.set_printoptions(precision=2)
plt_act_trio()
# UNQ_C1
# GRADED CELL: my_softmax
def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ez=np.exp(z)
    return ez/np.sum(ez)
z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf=tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")
# BEGIN UNIT TEST  
test_my_softmax(my_softmax)
# END UNIT TEST  
plt.close("all")
plt_softmax(my_softmax)
# load dataset
X, y = load_data()
print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
print(f'The shape of X is: {str(X.shape)}')
print(f'The shape of y is: {str(y.shape)}')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
M,n =X.shape
fig, axes=plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
#fig.tight_layout(pad=0.5)
widgvis(fig)
for i, ax in enumerate(axes.flat):
    # Select random indices 
    random_index=np.random.randint(M)
     # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped=X[random_index].reshape(20, 20).T
    # Display the image
    ax.imshow(X_random_reshaped, cmap="grey")
    # Display the label above the image
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()
# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234)
model=Sequential(
    [
        Input(shape=(400,)),
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="linear"),
    ], name="my_model"
)
model.summary()
[layer1, layer2, layer3]=model.layers
W1, b1=layer1.get_weights()
W2, b2=layer2.get_weights()
W3, b3=layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer=optimizers.Adam(learning_rate=0.001)
)
history=model.fit(
    X,y,
    epochs=100
)
#plot_loss_tf(history)
image_of_two = X[1015]
display_digit(image_of_two)
prediction=model.predict(image_of_two.reshape(1, 400))
print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")
prediction_p=tf.nn.softmax(prediction)
print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")
yhat = np.argmax(prediction_p)
print(f"np.argmax(prediction_p): {yhat}")
m, n=X.shape
fig, axes=plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index=np.random.randint(m)
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped=X[random_index].reshape((20,20)).T
    # Display the image
    ax.imshow(X_random_reshaped, cmap="grey")
    # Predict using the Neural Network
    prediction=model.predict(X[random_index].reshape(1, 400))
    prediction_p=tf.nn.softmax(prediction)
    yhat=np.argmax(prediction_p)
    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]}, {yhat}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()
print( f"{display_errors(model,X,y)} errors out of {len(X)} images")
np.set_printoptions(precision=2)
tf.get_logger().setLevel("ERROR")
# Load the dataset from the text file
data=np.loadtxt("./data/data_w3_ex1.csv", delimiter=",")
# Split the inputs and outputs into separate arrays
x=data[:, 0]
y=data[:, 1]
# Convert 1-D arrays into 2-D because the commands later will require it
x=np.expand_dims(x, axis=1)
y=np.expand_dims(y, axis=1)
print(f"the shape of the inputs x is: {x.shape}")
print(f"the shape of the targets y is: {y.shape}")
# Plot the entire dataset
utils.plot_dataset(x=x, y=y, title="input vs. target")
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_=train_test_split(x, y, test_size=0.40, random_state=1)
# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test=train_test_split(x_, y_, test_size=0.50, random_state=1)
# Delete temporary variables
del x_, y_
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")
utils.plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target")
# Initialize the class
scaler_linear=StandardScaler()
# Compute the mean and standard deviation of the training set then transform it
X_train_scaled=scaler_linear.fit_transform(x_train)
print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")
# Plot the results
utils.plot_dataset(x=X_train_scaled, y=y_train, title="scaled input vs. target")
linear_model=LinearRegression()
linear_model.fit(X_train_scaled, y_train)
# Feed the scaled training set and get the predictions
yhat=linear_model.predict(X_train_scaled)
# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat)/2}")
# for-loop implementation
total_squared_error = 0
for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                 
mse = total_squared_error / (2*len(yhat))
sei=(yhat-y_train)**2
total=np.sum(sei)
mse=total/(2*len(yhat))
print(f"training MSE (for-loop implementation): {mse.squeeze()}")
# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled=scaler_linear.transform(x_cv)
print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")
# Feed the scaled cross validation set
yhat=linear_model.predict(X_cv_scaled)
# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat)/2}")
# Instantiate the class to make polynomial features
poly=PolynomialFeatures(degree=2, include_bias=False)
# Compute the number of features and transform the training set
X_train_mapped=poly.fit_transform(x_train)
# Preview the first 5 elements of the new training set. Left column is `x` and right column is `x^2`
# Note: The `e+<number>` in the output denotes how many places the decimal point should 
# be moved. For example, `3.24e+03` is equal to `3240`
print(X_train_mapped[:5])
# Instantiate the class
scaler_poly=StandardScaler()
# Compute the mean and standard deviation of the training set then transform it
X_train_mapped_scaled=scaler_poly.fit_transform(X_train_mapped)
# Preview the first 5 elements of the scaled training set.
print(X_train_mapped_scaled[:5])
# Initialize the class
model=LinearRegression()
# Train the model
model.fit(X_train_mapped_scaled, y_train)
# Compute the training MSE
yhat=model.predict(X_train_mapped_scaled)
yhat = model.predict(X_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")
# Add the polynomial features to the cross validation set
X_cv_mapped=poly.transform(x_cv)
# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_mapped_scaled=scaler_poly.transform(X_cv_mapped)
# Compute the cross validation MSE
yhat=model.predict(X_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")
# Initialize lists containing the lists, models, and scalers
train_mses = []
cv_mses = []
models = []
scalers = []
for degree in range(1, 11):
    # Add polynomial features to the training set
    poly=PolynomialFeatures(degree, include_bias=False)
    X_train_mapped=poly.fit_transform(x_train)
    # Scale the training set
    scaler_poly=StandardScaler()
    X_train_mapped_scaled=scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    # Create and train the model
    model=LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)
    # Compute the training MSE
    yhat=model.predict(X_train_mapped_scaled)
    train_mse=mean_squared_error(y_train, yhat)/2
    train_mses.append(train_mse)
    # Add polynomial features and scale the cross validation set
    X_cv_mapped=poly.transform(x_cv)
    X_cv_mapped_scaled=scaler_poly.transform(X_cv_mapped)
    # Compute the cross validation MSE
    yhat=model.predict(X_cv_mapped_scaled)
    cv_mse=mean_squared_error(y_cv, yhat)/2
    cv_mses.append(cv_mse)
# Plot the results
degrees=range(1,11)
utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")
# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree=np.argmin(cv_mses)+1
print(f"Lowest CV MSE is found in the model with degree={degree}")
# Add polynomial features to the test set
poly=PolynomialFeatures(degree, include_bias=False)
X_test_mapped=poly.fit_transform(x_test)
# Scale the test set
X_test_mapped_scaled=scalers[degree-1].transform(X_test_mapped)
# Compute the test MSE
yhat=models[degree-1].predict(X_test_mapped_scaled)
test_mse=mean_squared_error(y_test, yhat)/2
print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
# Add polynomial features
degree=1
poly=PolynomialFeatures(degree, include_bias=False)
X_train_mapped=poly.fit_transform(x_train)
X_cv_mapped=poly.transform(x_cv)
X_test_mapped=poly.transform(x_test)
# Scale the features using the z-score
scaler=StandardScaler()
X_train_mapped_scaled=scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled=scaler.transform(X_cv_mapped)
X_test_mapped_scaled=scaler.transform(X_test_mapped)
# Initialize lists that will contain the errors for each model
nn_train_mses=[]
nn_cv_mses=[]
# Build the models
nn_models=utils.build_models()
# Loop over the the models
for model in nn_models:
    # Setup the loss and optimizer
    model.compile(
        loss="mse", #losses.MeanSquaredError()
        optimizer=optimizers.Adam(learning_rate=0.1)
    )
    print(f"Training {model.name}...")
    # Train the model
    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=300,
        verbose=0
    )
    print("Done!\n")
    # Record the training MSEs
    yhat=model.predict(X_train_mapped_scaled)
    train_mse=mean_squared_error(y_train, yhat)/2
    nn_train_mses.append(train_mse)
    # Record the cross validation MSEs 
    yhat=model.predict(X_cv_mapped_scaled)
    cv_mse=mean_squared_error(y_cv, yhat)/2
    nn_cv_mses.append(cv_mse)
# print results
print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, "+
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
    )
# Select the model with the lowest CV MSE
model_num=3
# Compute the test MSE
yhat=nn_models[model_num-1].predict(X_test_mapped_scaled)
test_mse=mean_squared_error(y_test, yhat)/2
print(f"Selected Model: {model_num}")
print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
# Load the dataset from a text file
data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')
# Split the inputs and outputs into separate arrays
x_bc=data[:,:-1]
y_bc=data[:, -1]
# Convert y into 2-D because the commands later will require it (x is already 2-D)
y_bc=np.expand_dims(y_bc, axis=1)
print(f"the shape of the inputs x is: {x_bc.shape}")
print(f"the shape of the targets y is: {y_bc.shape}")
utils.plot_bc_dataset(x=x_bc, y=y_bc, title="x1 vs. x2")
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_, y_bc_train, y_=train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)
# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test=train_test_split(x_, y_, test_size=0.50, random_state=1)
# Delete temporary variables
del x_, y_
print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")
# Sample model output
probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])
# Apply a threshold to the model output. If greater than 0.5, set to 1. Else 0.
predictions=np.where(probabilities>=0.5, 1, 0)
# Ground truth labels
ground_truth = np.array([1, 1, 1, 1, 1])
# Initialize counter for misclassified data
misclassified=0
# Get number of predictions
num_predictions = len(predictions)
# Loop over each prediction
for i in range(num_predictions):
    # Check if it matches the ground truth
    if predictions[i]!=ground_truth[i]:
         # Add one to the counter if the prediction is wrong
        misclassified+=1
fraction_error=misclassified/num_predictions
print(f"probabilities: {probabilities}")
print(f"predictions with threshold=0.5: {predictions}")
print(f"targets: {ground_truth}")
print(f"fraction of misclassified data (for-loop): {fraction_error}")
print(f"fraction of misclassified data (with np.mean()): {np.mean(predictions!=ground_truth)}")
# Initialize lists that will contain the errors for each model
nn_train_error = []
nn_cv_error = []
# Build the models
models_bc = utils.build_models()
# Set the threshold for classification
threshold=0.5
# Loop over each model
for model in models_bc:
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(learning_rate=0.01)
    )
    print(f"Training {model.name}...")
    # Train the model
    model.fit(
        x_bc_train, y_bc_train,
        epochs=200,
        verbose=0
    )
    print("Done!\n")
    # Record the fraction of misclassified examples for the training set
    yhat=model.predict(x_bc_train)
    yhat=tf.math.sigmoid(yhat)
    yhat=np.where(yhat>=threshold, 1, 0)
    train_error=np.mean(yhat!=y_bc_train)
    nn_train_error.append(train_error)
    # Record the fraction of misclassified examples for the cross validation set
    yhat=model.predict(x_bc_cv)
    yhat=tf.math.sigmoid(yhat)
    yhat=np.where(yhat>=threshold, 1, 0)
    cv_error=np.mean(yhat!=y_bc_cv)
    nn_cv_error.append(cv_error)
# Print the result
for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
        f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
        )
# Select the model with the lowest error
model_num=3
# Compute the test error
yhat=models_bc[model_num-1].predict(x_bc_test)
yhat=tf.math.sigmoid(yhat)
yhat=np.where(yhat>=0.5, 1, 0)
nn_test_error=np.mean(yhat!=y_bc_test)
print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")
# Split the dataset into train, cv, and test
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data1.csv')
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
# Preview the first 5 rows
print(f"first 5 rows of the training inputs (1 feature):\n {x_train[:5]}\n")
# Instantiate the regression model class
model=LinearRegression()
# Train and plot polynomial regression models
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)
# Train and plot polynomial regression models. Bias is defined lower.
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
# Preview the first 5 rows
print(f"first 5 rows of the training inputs (2 features):\n {x_train[:5]}\n")
# Instantiate the model class
model = LinearRegression()
# Train and plot polynomial regression models. Dataset used has two features.
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)
# Define lambdas to plot
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]
# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
# Define lambdas to plot
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')
# Preview the first 5 rows
print(f"first 5 rows of the training set with 2 features:\n {x_train[:5]}\n")
# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data3.csv')
# Preview the first 5 rows
print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {x_train[:5]}\n")
# Define the model
model = LinearRegression()
# Define properties of the 2 datasets
file1 = {'filename':'data/c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
file2 = {'filename':'data/c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
files = [file1, file2]
# Train and plot for each dataset
utils.train_plot_diff_datasets(model, files, max_degree=4, baseline=250)
# Prepare the dataset
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data4.csv')
print(f"the shape of the entire training set (input) is: {x_train.shape}")
print(f"the shape of the entire training set (target) is: {y_train.shape}\n")
print(f"the shape of the entire cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the entire cross validation set (target) is: {y_cv.shape}\n")
# Instantiate the model class
model = LinearRegression()
# Define the degree of polynomial and train the model using subsets of the dataset.
utils.train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
tf.keras.backend.set_floatx("float64")
# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)
#split the data using sklearn routine 
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
fig, ax=plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "--", color="orangered", label="Y_ideal", lw=1)
ax.set_title("Training, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("Y")
ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_test, y_test, color=dlc["dlblue"],    label="test")
ax.legend(loc="upper left")
plt.show()
# UNQ_C1
# GRADED CELL: eval_mse
def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m=len(y)
    err=(yhat-y)**2
    return np.sum(err)/(2*m)
y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
eval_mse(y_hat, y_tmp)
# BEGIN UNIT TEST
test_eval_mse(eval_mse)   
# END UNIT TEST
# create a model in sklearn, train on training data
degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)
# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)
# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")
# plot predictions over data range 
x=np.linspace(0, int(X.max()), 100) # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)
plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
# Generate  data
X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)
#split the data using sklearn routine 
X_train, x_, y_train, y_=train_test_split(X, y, test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test=train_test_split(x_, y_, test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
fig, ax=plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_label", lw=1)
ax.set_title("Training, CV, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_cv, y_cv, color=dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
ax.legend(loc='upper left')
plt.show()
max_degree = 9
err_train = np.zeros(max_degree)    
err_cv = np.zeros(max_degree)    
x=np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100, max_degree))  #columns are lines to plot
for degree in range(max_degree):
    lmodel = lin_model(degree+1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:,degree] = lmodel.predict(x)
optimal_degree = np.argmin(err_cv)+1
plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                   err_train, err_cv, optimal_degree, max_degree)
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)    
err_cv = np.zeros(num_steps)       
x = np.linspace(0,int(X.max()),100) 
y_pred = np.zeros((100,num_steps))  #columns are lines to plot
for i in range(num_steps):
    lambda_= lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:,i] = lmodel.predict(x)
optimal_reg_idx = np.argmin(err_cv) 
plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
# Generate and split data set
X, y, centers, classes, std = gen_blobs()
# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
print("X.shape", X.shape, "X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)
plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)
# UNQ_C2
# GRADED CELL: eval_cat_err
def eval_cat_err(y, yhat):
    """ 
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)             
    """
    m=len(y)
    incorrect = sum(int((y[i]!=yhat[i])) for i in range(m))
    return incorrect/m
y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )
# BEGIN UNIT TEST  
test_eval_cat_err(eval_cat_err)
# END UNIT TEST
# UNQ_C3
# GRADED CELL: model
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ### 
    Dense(units=120,activation='relu'),    
    Dense(units=40,activation='relu'),
    Dense(units=6,activation='linear')
        
        ### END CODE HERE ### 

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01),
    ### END CODE HERE ### 
)
# BEGIN UNIT TEST
model.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
model.summary()
# END UNIT TEST
#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")
training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")
tf.random.set_seed(1234)
model=Sequential(
    [
        Dense(6, activation="relu"),
        Dense(6, activation="linear")
    ], name="simple"
)
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01)
)
model.fit(
    X_train, y_train,
    epochs=1000
)
# BEGIN UNIT TEST
model.summary()
# END UNIT TEST
#make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")
training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
tf.random.set_seed(1234)
model_r=Sequential(
    [
        Dense(120, activation="relu", kernel_regularizer=regularizers.l2(0.1)),
        Dense(40, activation="relu", kernel_regularizer=regularizers.l2(0.1)),
        Dense(6, activation="linear")
    ], name= None
)
model_r.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01)
)
# BEGIN UNIT TEST
model_r.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
# BEGIN UNIT TEST
model_r.summary()
# END UNIT TEST
#make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_r, X_train,y_train, classes, X_cv, y_cv, suptitle="Regularized")
training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")
from utils import *
_ = plot_entropy()
X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
#For instance, the first example
print(X_train[0])
def entropy(p):
    return 0 if p in [0, 1] else -p*np.log2(p)-(1-p)*np.log2(1-p)
print(entropy(0.5))
def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices=[]
    right_indices=[]
    for i,x in enumerate(X):
        left_indices.append(i) if x[index_feature]==1 else right_indices.append(i)
    return left_indices, right_indices
print(split_indices(X_train, 0))
def weighted_entropy(X, y, left_indices, right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left=len(left_indices)/len(X)
    w_right=len(right_indices)/len(X)
    p_left=sum(y[left_indices])/len(left_indices)
    p_right=sum(y[right_indices])/len(right_indices)
    return w_left*entropy(p_left)+w_right*entropy(p_right)
left_indices, right_indices = split_indices(X_train, 0)
print(weighted_entropy(X_train, y_train, left_indices, right_indices))
def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    return entropy(sum(y)/len(y))-weighted_entropy(X, y, left_indices, right_indices)
print(information_gain(X_train, y_train, left_indices, right_indices))
for i,  feature_name in enumerate(["Ear shape", "Face shape", "Wiskers"]):
    left_indices, right_indices=split_indices(X_train, i)
    i_gain=information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=1, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
RANDOM_STATE=55 ## We will pass it to every sklearn call so we ensure reproducibility
# Load the dataset using pandas
df=pd.read_csv("heart.csv")
print(df.head())
cat_variables=["Sex",
"ChestPainType",
"RestingECG",
"ExerciseAngina",
"ST_Slope"
]
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df=pd.get_dummies(data=df,
prefix=cat_variables,
columns=cat_variables)
print(df.head())
features=[x for x in df.columns if x not in "HeartDisease"]
print(len(features))
#help(train_test_split)
X_train, X_val, y_train, y_val=train_test_split(df[features], df["HeartDisease"], train_size=0.8, random_state=RANDOM_STATE)
# We will keep the shuffle = True since our dataset has not any time dependency.
print(f'train samples: {len(X_train)} validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
accuracy_list_train=[]
accuracy_list_val=[]
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model=DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=RANDOM_STATE).fit(X_train, y_train)
    prediction_train=model.predict(X_train) ## The predicted values for the train dataset
    prediction_val=model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train=accuracy_score(prediction_train, y_train)
    accuracy_val=accuracy_score(prediction_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(["train", "validation"])
plt.show()
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()
decision_tree_model=DecisionTreeClassifier(min_samples_split=50,
max_depth=4,
random_state=RANDOM_STATE).fit(X_train, y_train)
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")
n_estimators_list = [10,50,100,500]
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model=RandomForestClassifier(min_samples_split=min_samples_split,
    random_state=RANDOM_STATE).fit(X_train, y_train)
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
random_forest_model=RandomForestClassifier(n_estimators=100,
max_depth=16,
min_samples_split=10).fit(X_train, y_train)
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")
n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]
xgb_model=XGBClassifier(n_estimator=500, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds=10)
xgb_model.best_iteration'''
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy=0
    ### START CODE HERE ###
    if len(y)!=0:
        p_1=np.sum(y)/len(y)
        entropy = -p_1*np.log2(p_1)-(1-p_1)*np.log2(1-p_1) if p_1 not in [1, 0] else 0
    return entropy
# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"

print("Entropy at root node: ", compute_entropy(y_train)) 

# UNIT TESTS
compute_entropy_test(compute_entropy)
# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    for i in node_indices:
        left_indices.append(i) if X[i, feature]==1 else right_indices.append(i)
    return left_indices, right_indices
# Case 1

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices, left_indices, right_indices, feature)

print()

# Case 2

root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)

print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

# UNIT TESTS    
split_dataset_test(split_dataset)
# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):

    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    left_indices, right_indices=split_dataset(X, node_indices, feature)
    X_node, y_node=X[node_indices], y[node_indices]
    X_left, y_left=X[left_indices], y[left_indices]
    X_right, y_right=X[right_indices], y[right_indices]
    # You need to return the following variables correctly
    information_gain = 0
    H_node=compute_entropy(y_node)
    H_left=compute_entropy(y_left)
    H_right=compute_entropy(y_right)
    w_left=len(X_left)/len(X_node)
    w_right=len(X_right)/len(X_node)
    return H_node-(w_left*H_left+w_right*H_right)
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# UNIT TESTS
compute_information_gain_test(compute_information_gain)
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    # You need to return the following variables correctly
    best_feature = -1
    max_info_gain=0
    for i in range(num_features):
        information_gain=compute_information_gain(X, y, node_indices, i) 
        if information_gain>max_info_gain:
            max_info_gain=information_gain
            best_feature=i
    return best_feature
best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

# UNIT TESTS
get_best_split_test(get_best_split)
# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 
     # Maximum depth reached - stop splitting
    if current_depth==max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, f"{branch_name} leaf node with indices", node_indices)
        return 
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature=get_best_split(X, y, node_indices)
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    # Split the dataset at the best feature
    left_indices, right_indices=split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "right", max_depth, current_depth+1)
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)