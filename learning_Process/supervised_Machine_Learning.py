# Linear regression in one variable
import numpy as np   # a popular library for scientific computing
import math, copy
import matplotlib.pyplot as plt # a popular library for scientific computing
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w, zscore_normalize_features, run_gradient_descent_feng
from lab_utils_common import dlc, plot_data, draw_vthresh,plt_tumor_data, sigmoid, compute_cost_logistic
from plt_one_addpt_onclick import plt_one_addpt_onclick
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
from plt_quad_logistic import plt_quad_logistic, plt_prob
from plt_overfit import overfit_example, output
from utils import *
'''from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl, plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0]) #features 
y_train = np.array([300, 500]) # target value
print(f"x_train={x_train}")
print(f"y_train={y_train}")
# m is the number of training examples
print(f"x_train.shape={x_train.shape}")  # shape parameter. x_train.shape returns a python tuple with an entry for each dimension. x_train.shape[0] is the length of the array
# m=x_train.shape[0]
# m=len(x_train)
# print(f"the no. of training set is: {m}")
i=1
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# plot the data points
plt.scatter(x_train, y_train, marker="x", c="r")
# Set the title
plt.title("Housing prices")
# set the y axis label
plt.ylabel("Price (in 1000s of dollars)")
# set the x axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
w=200
b=100
print(f"w: {w}")
print(f"b: {b}")
def compute_model_output(x, w, b):
  """
  Computes the prediction of a linear model
  Args:
    x (ndarray (m,)): Data, m examples 
    w,b (scalar)    : model parameters  
  Returns
    y (ndarray (m,)): target values
  """
  m=x.shape[0]
  f_wb=np.zeros(m)  # returns a new a array filled with zeros of shape m
  for i in range(m):
      f_wb[i]=w*x[i]+b
  return f_wb
tmp_f_wb=compute_model_output(x_train, w, b)
# plot our model prediction
plt.plot(x_train, tmp_f_wb, c="b", label="Our prediction")   #c means color
# plot the data points
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
x_i=1.2
cost_1200sqft=w*x_i+b
print(f"${cost_1200sqft:.0f} thousand dollars")
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m=len(x)
    cost_sum=0
    for i in range(m):
      f_wb=w*x[i]+b
      cost_sum+=(f_wb-y[i])**2
    return (1/(2*m))*cost_sum
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b """
    m=len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
      f_wb=w*x[i]+b
      dj_dw+=(f_wb-y[i])*x[i]
      dj_db+=(f_wb-y[i])
    return dj_dw/m , dj_db/m 
def gradient_descent(x, y, w, b, alpha, num_iters): 
  """
  Performs gradient descent to fit w,b. Updates w,b by taking 
  num_iters gradient steps with learning rate alpha
  
  Args:
    x (ndarray (m,))  : Data, m examples 
    y (ndarray (m,))  : target values
    w_in,b_in (scalar): initial values of model parameters  
    alpha (float):     Learning rate
    num_iters (int):   number of iterations to run gradient descent
    cost_function:     function to call to produce cost
    gradient_function: function to call to produce gradient
    
  Returns:
    w (scalar): Updated value of parameter after running gradient descent
    b (scalar): Updated value of parameter after running gradient descent
    J_history (List): History of cost values
    p_history (list): History of parameters [w,b] 
    """
  # An array to store cost J and w's at each iteration primarily for graphing later
  j_history=[]
  p_history=[]
  # Update Parameters
  for i in range(num_iters):
      # Calculate the gradient and update the parameters using gradient_function
    dj_dw, dj_db=compute_gradient(x, y, w, b)
    b=b-alpha*dj_db
    w=w-alpha*dj_dw
    if i<100000:
      j_history.append(compute_cost(x,y,w,b))
      p_history.append([w, b])
    # Print cost every at intervals 10 times or as many iterations if < 10
    if i% math.ceil(num_iters/10) == 0:
        print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
              f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
              f"w: {w: 0.3e}, b:{b: 0.5e}")
  return w, b, j_history, p_history
#print(compute_cost(x_train, y_train, w, b))
# initialize parameters
w_init=0
b_init=0
# some gradient descent settings
iteration=10000
tmp_alp=1.0e-2
w_final,b_final, j_hist, p_hist= gradient_descent(x_train, y_train, w_init, b_init, tmp_alp, iteration)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
plt_intuition(x_train,y_train)
plt.close('all') 
fig, ax, dyn_items=plt_stationary(x_train, y_train)
updater=plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
fig, (ax1, ax2)=plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(j_hist[:100])
ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt.show()
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)
plt.show()
# numpy routines which allocate memory and fill arrays with value
import numpy as np
import time
a = np.zeros(4); print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a=np.zeros((4,)); print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a=np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a=np.arange(4.); print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a=np.random.rand(4); print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)
#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")
#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")
#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)
# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)
# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)
# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)
# access all elements
c = a[:];         print("a[:]     = ", c)
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a  
b=-a
print(f"b = -a        : {b}")
b=np.sum(a)
print(f"b = np.sum(a) : {b}")
b= np.mean(a)
print(f"b = np.mean(a): {b}")
b=a**2
print(f"b = a**2      : {b}")
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
a=np.array([1, 2, 3, 4])
b=a*5
print(f"b = 5 * a : {b}")
def my_dot(a, b):
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    return sum((a[i] * b[i]) for i in range(a.shape[0]))
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c=np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
np.random.seed(1)
a=np.random.rand(10000000)      # very large arrays
b=np.random.rand(10000000)
tic=time.time()    # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time
print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")
tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time
print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")
del(a); del(b)  #remove these big arrays from  memory
# show common Course 1 example
X=np.array([[1], [2], [3], [4]])
w=np.array([2])
c = np.dot(X[1], w)
print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
a=np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")         
a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 
a=np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}") 
a=np.array([[5], [4], [3]]); print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")
#vector indexing operations on matrices
a=np.arange(6).reshape(-1, 2)   #reshape is a convenient way to form matrices
print(f"a.shape: {a.shape}, \na= {a}")
#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")
#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")
print("a[0, 2:7:1] = ", a[0, 2:7], ",  a[0, 2:7:1].shape =", a[0, 2:7].shape, "a 1-D array")
#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")
# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)
# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
import numpy as np
import matplotlib.pyplot as plt
import copy, math
X_train=np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852,	2,	1, 35]])
y_train = np.array([460, 232, 178])
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
def predict_single_loop(x, w, b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i
    return p + b
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    return np.dot(w, x)+b
# get a row from our training data
x_vec=X_train[0]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m=X.shape[0]
    cost=0
    for i in range(m):
      f_wb_i=np.dot(X[i], w)+b
      cost+=(f_wb_i-y[i])**2
    return (1/(2*m))*cost
# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n=X.shape   #(number of examples, number of features)
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
      err=(np.dot(X[i], w)+b)-y[i]
      for j in range(n):
        dj_dw[j]+=err*X[i, j]
      dj_db+=err
    return dj_dw/m, dj_db/m
#Compute and display gradient 
tmp_dj_dw, tmp_dj_db=compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
    """
    J_history=[]
    w=copy.deepcopy(w_in)   # avoid modifying global w within function
    b=b_in
    for i in range(num_iters):
       # Calculate the gradient and update the parameters
      dj_dw, dj_db=compute_gradient(X, y, w, b)
      w-=alpha*dj_dw
      b-=alpha*dj_db
      # Update Parameters using w, b, alpha and gradient
      # Save cost J at each iteration
      if i<100000:      # prevent resource exhaustion 
          J_history.append(compute_cost(X, y, w, b))
      # Print cost every at intervals 10 times or as many iterations if < 10
      if i% math.ceil(num_iters / 10) == 0:
          print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history #return final w,b and J history for graphing
initial_w=np.zeros_like(w_init)
initial_b=0
# some gradient descent settings
iterations = 1000
alpha=5.0e-7
# run gradient descent
w_final, b_final, J_hist=gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m=X_train.shape[0]
for i in range(m):
  print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
fig, (ax1, ax2)=plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100+np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
np.set_printoptions(precision=2)
# load the dataset
X_train, y_train=load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
fig, ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
  ax[i].scatter(X_train[:,i], y_train)
  ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()
#set alpha to 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
plot_cost_i_w(X_train, y_train, hist)
#set alpha to 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
plot_cost_i_w(X_train, y_train, hist)
#set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train,y_train,hist)
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu=np.mean(X, axis=0)    # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma=np.std(X, axis=0)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      
    return (X_norm, mu, sigma)
mu     = np.mean(X_train,axis=0)  
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma   
fig, ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')
ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')
ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")
plt.show()
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1)
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict=np.dot(w_norm, x_house_norm)+b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
plt_equal_scale(X_train, X_norm, y_train)
# create target data # reduced display precision on numpy arrays
x=np.arange(0, 20, 1)
y=1+x**2
X=x.reshape(-1,1)
model_w, model_b=run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
# Engineer features 
X=x**2   #<-- added engineered feature
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
X = np.c_[x, x**2, x**3]   
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
X_features = ['x','x^2','x^3']
fig, ax=plt.subplots(1, 3, figsize=(12,3), sharey=True)
for i in range(len(ax)):
  ax[i].scatter(X[:, i], y)
  ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")
y=x**2
# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
y=np.cos(x/2)
X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 
model_w,model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha = 1e-1)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
scaler=StandardScaler()
X_norm=scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
sgdr=SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
b_norm=sgdr.intercept_
w_norm=sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")
# make a prediction using sgdr.predict()
y_pred_sgd=sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")
print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")
fig, ax=plt.subplots(1, 4, figsize=(12,3),sharey=True)
for i in range(len(ax)):
  ax[i].scatter(X_train[:, i], y_train, label = 'target')
  ax[i].set_xlabel(X_features[i])
  ax[i].scatter(X_train[:, i], y_pred, color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value
linear_model=LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")
y_pred=linear_model.predict(X_train.reshape(-1,1))
print("Prediction on training set:", y_pred)
X_test=np.array([1200])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test.reshape(-1, 1))[0]:0.2f}")
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
linear_model.fit(X_train, y_train)
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")
x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_pred=linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_pred*1000:0.2f}")
from utils import *
from public_tests import *
# load the dataset
x_train, y_train = load_data()
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
# UNQ_C1
# GRADED FUNCTION: compute_cost
def compute_cost(x, y, w, b):
  """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
  m=x.shape[0]
  cost=0
  for  i in range(m):
    f_wb=w*x[i]+b
    cost+=(f_wb-y[i])**2
  return cost/(2*m)
# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')
# Public tests
from public_tests import *
compute_cost_test(compute_cost)
# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
      f_wb=w*x[i]+b
      dj_dw+=(f_wb-y[i])*x[i]
      dj_db+=(f_wb-y[i])
    return dj_dw/m, dj_db/m
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)
compute_gradient_test(compute_gradient)
# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)
print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
  """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
  m=x.shape[0]
    # An array to store cost J and w's at each iteration — primarily for graphing later
  J_history = []
  w_history = []
  w = copy.deepcopy(w_in)  #avoid modifying global w within function
  b = b_in
  for i in range(num_iters):
    dj_dw, dj_db=compute_gradient(x, y, w, b)
    w-=alpha*dj_dw
    b-=alpha*dj_db
    if i<10000:
      cost=compute_cost(x, y, w, b)
      J_history.append(cost)
    if i% math.ceil(num_iters/10)==0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
  return w, b, J_history, w_history
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w=0.
initial_b=0.
# some gradient descent settings
iterations=1500
alpha=0.01
w,b,_,_=gradient_descent(x_train ,y_train, initial_w, initial_b,  alpha, iterations)
print("w,b found by gradient descent:", w, b)
m=x_train.shape[0]
predicted=np.zeros(m)
for i in range(m):
  predicted[i]=w*x_train[i]+b
plt.plot(x_train, predicted, c = "b")
# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))
predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])
pos=y_train==1
neg=y_train==0
fig, ax=plt.subplots(1, 2, figsize=(8,3))
#plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c='red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', edgecolors=dlc["dlblue"],lw=3)
ax[0].set_ylim(-0.08, 1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()
#plot 2, two variables
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()
w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)
plt.show()
# Input is an array. 
input_array = np.array([1,2,3])
exp_array=np.exp(input_array)
print("Input to exp:", input_array)
print("Output of exp:", exp_array)
# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)
print("Input to exp:", input_val)
print("Output of exp:", exp_val)'''
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    return 1/(1+np.exp(-z))
'''# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)
# Use the function implemented above to get the sigmoid values
y=sigmoid(z_tmp)
# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])
# Plot z vs sigmoid(z)
fig, ax=plt.subplots(1,1, figsize=(5,3))
ax.plot(z_tmp, y, c="b")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()
x_train=np.array([0., 1, 2, 3, 4, 5])
y_train=np.array([0, 0, 0, 1, 1, 1])
w_in=np.zeros(1)
b_in=0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
plt.show()
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()
# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)
fig, ax=plt.subplots(1,1, figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()
# Choose values between 0 and 6
x0 = np.arange(0,6)
x1 = 3 - x0
fig, ax=plt.subplots(1,1,figsize=(5, 4))
# Plot the decision boundary
ax.plot(x0, x1, c="b")
ax.axis([0, 4, 0, 3.5])
# Fill the region below the line
ax.fill_between(x0, x1, alpha=0.2)
# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
soup_bowl()
x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
plt.show()
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()
plt_two_logistic_loss_curves()
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)
plt.show()
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)
# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m=X.shape[0]
    cost=0
    for i in range(m):
      z_i=np.dot(w, X[i])+b
      f_wb_i=sigmoid(z_i)
      cost+=-y[i]*(np.log(f_wb_i))-(1-y[i])*(np.log(1-f_wb_i))
    return cost/m
w_tmp = np.array([1,1])
b_tmp = -3        
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
# Choose values between 0 and 6
x0 = np.arange(0,6)
# Plot the two decision boundaries
x1=3-x0
x1_other=4-x0
fig, ax=plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0, x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0, x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])
# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()
w_array1 = np.array([1,1])
b_1 = -3
b_2 = -4
print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array1, b_2))
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m, n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
      f_wb_i=1/(1+np.exp(-(np.dot(w, X[i])+b)))
      for j in range(n):
        dj_dw[j]+=(f_wb_i-y[i])*X[i, j]
      dj_db+=f_wb_i-y[i]
    return dj_db/m, dj_dw/m
w_tmp = np.array([2.,3.])
b_tmp = 1
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_train, y_train, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
      # Calculate the gradient and update the parameters
      dj_db, dj_dw=compute_gradient_logistic(X, y, w, b)
      # Update Parameters using w, b, alpha and gradient
      w-=alpha*dj_dw
      b-=alpha*dj_db
      # Save cost J at each iteration
      if i<100000:  # prevent resource exhaustion 
        J_history.append(compute_cost_logistic(X, y, w, b))
      # Print cost every at intervals 10 times or as many iterations if < 10
      if i%math.ceil(num_iters/10)==0:
        print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history         #return final w,b and J history for graphing
w_tmp=np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000
w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability 
plt_prob(ax, w_out, b_out)
# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)
# Plot the decision boundary
x0=-b_out/w_out[0]
x1=-b_out/w_out[1]
ax.plot([0, x0], [x1, 0], c=dlc["dlblue"], lw=1)
plt.show()
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )
plt.show()
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(X,y)
y_pred=lr_model.predict(X)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X, y))
plt.close("all")
display(output)
fit = overfit_example(False)
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    m,n=X.shape
    cost=0
    for i in range(m):
      f_wb=np.dot(w, X[i])+b
      cost+=(f_wb-y[i])**2
    cost/=(2*m)
    reg_cost = sum(w[j]**2 for j in range(n))
    reg_cost*=(lambda_/(2*m))
    cost+=reg_cost
    return cost
np.random.seed(1)
X_tmp=np.random.rand(5,6)
y_tmp=np.array([0,1,0,1,0])
w_tmp=np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp=0.5
lambda_tmp=0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost:", cost_tmp)
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    m, n=X.shape
    cost = sum(
        -y[i] * np.log(1 / (1 + np.exp(-(np.dot(w, X[i]) + b))))
        - (1 - y[i]) * np.log(1 - (1 / (1 + np.exp(-(np.dot(w, X[i]) + b)))))
        for i in range(m)
    )
    cost/=m
    reg_cost=sum(w[j]**2 for j in range(n))
    reg_cost*=(lambda_/(2*m))
    cost+=reg_cost
    return cost
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost:", cost_tmp)
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
      for j in range(n):
        dj_dw[j]+=((np.dot(w, X[i])+b)-y[i])*X[i,j]
      dj_db+=((np.dot(w, X[i])+b)-y[i])
    dj_dw/=m
    dj_db/=m
    for j in range(n):
      dj_dw[j]+=(lambda_/m)*w[j]
    return dj_db, dj_dw
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
      for j in range(n):
        dj_dw[j]+=((1/(1+np.exp(-(np.dot(w, X[i])+b))))-y[i])*X[i,j]
      dj_db+=((1/(1+np.exp(-(np.dot(w, X[i])+b))))-y[i])
    dj_dw/=m
    dj_db/=m
    for j in range(n):
      dj_dw[j]+=(lambda_/m)*w[j]
    return dj_db, dj_dw
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}")'''
X_train, y_train=load_data("data/ex2data1.txt")
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))
print(f'The shape of X_train is: {str(X_train.shape)}')
print(f'The shape of y_train is: {str(y_train.shape)}')
print ('We have m = %d training examples' % (len(y_train)))
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
# UNQ_C1
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    return 1/(1+np.exp(-z))
print(f"sigmoid(0) = {str(sigmoid(0))}")
print(f"sigmoid([ -1, 0, 1, 2]) = {str(sigmoid(np.array([-1, 0, 1, 2])))}")
# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)
# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """
    m, n=X.shape
    loss=0
    for i in range(m):
      f_wb=np.dot(w, X[i])+b
      g=sigmoid(f_wb)
      loss+=(-y[i]*np.log(g))-(1-y[i])*np.log(1-g)
    return loss/m
m, n = X_train.shape
# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))
# Compute and display cost with non-zero w
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)
print('Cost at test w,b: {:.3f}'.format(cost))
# UNIT TESTS
compute_cost_test(compute_cost)
# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n=X.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
      f_wb=np.dot(w, X[i])+b
      g=sigmoid(f_wb)
      for j in range(n):
        dj_dw[j]+=(g-y[i])*X[i, j]
      dj_db+=(g-y[i])
    return dj_db/m, dj_dw/m
# Compute and display gradient with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}' )
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )
# Compute and display cost and gradient with non-zero w
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)
print('dj_db at test_w:', dj_db)
print('dj_dw at test_w:', dj_dw.tolist())
# UNIT TESTS    
compute_gradient_test(compute_gradient)
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      
    Returns:
    w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    # number of training examples
    m=len(X)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history=[]
    w_history=[]
    for i in range (num_iters):
      dj_db, dj_dw=compute_gradient_reg(X, y, w_in, b_in, lambda_)
      w_in-=alpha*dj_dw
      b_in-=alpha*dj_db
      if i<100000:
        J_history.append(compute_cost_reg(X, y, w_in, b_in, lambda_))
      if i%math.ceil(num_iters/10)==0 or i==num_iters-1:
        w_history.append(w_in)
        print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing
np.random.seed(1)
'''
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8
# Some gradient descent settings
iterations = 100000
alpha = 0.001
w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, alpha, iterations, 0)
plot_decision_boundary(w, b, X_train, y_train)'''
# UNQ_C4
# GRADED FUNCTION: predict
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    m, n=X.shape
    p=np.zeros(m)
    # Loop over each example
    for i in range(m):
        # Loop over each feature
        # Add the corresponding term to z_wb
        z_wb=sum(X[i, j]*w[j] for j in range(n))
        # Add bias term 
        z_wb+=b
        # Calculate the prediction for this example
        f_wb=sigmoid(z_wb)
        # Apply the threshold
        p[i]=f_wb>=0.5
    return p
'''np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5
tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
# UNIT TESTS        
predict_test(predict)
#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))'''
# load dataset
X_train, y_train = load_data("data/ex2data2.txt")
# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))
# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()
print("Original shape of data:", X_train.shape)

X_mapped =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", X_mapped.shape)
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", X_mapped[0])
# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    m, n = X.shape
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    # You need to calculate this value
    reg_cost = 0.
    for j in range(n):
      reg_cost+=w[j]**2
    reg_cost*=lambda_/(2*m)
    reg_cost+=cost_without_reg
    return reg_cost
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
print("Regularized cost :", cost)
# UNIT TEST    
compute_cost_reg_test(compute_cost_reg)
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
      dj_dw[j]+=((lambda_/m)*w[j])
    return dj_db, dj_dw
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )
# UNIT TESTS    
compute_gradient_reg_test(compute_gradient_reg)     
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.
# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01                                         
# Some gradient descent settings
iterations = 10000
alpha = 0.01
w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, alpha, iterations, lambda_)
plot_decision_boundary(w, b, X_mapped, y_train)
#Compute accuracy on the training set
p = predict(X_mapped, w, b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

