import numpy as np
import matplotlib.pyplot as plt
from utils import *
# Initialize centroids
# K is the number of clusters
'''centroids=kMeans_init_centroids=(X, K)
for _ in range(interations):
    # Cluster assignment step: 
    # Assign each data point to the closest centroid. 
    # idx[i] corresponds to the index of the centroid 
    # assigned to example i
    idx=find_closest_centroids(X, centroids)
    # Move centroid step: 
    # Compute means based on centroid assignments
    centroids=compute_centroids(X, idx, k)
# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    # Set K
    K=centroids.shape[0]
    # You need to return the following variables correctly
    idx=np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance=[]
        for j in range(K):
            norm_ij=np.linalg.norm(X[i, :]-centroids[j, :], ord=2)
            distance.append(norm_ij)
        idx[i]=np.argmin(distance)
    return idx
# Load an example dataset that we will be using
X=load_data()
print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)
# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx = find_closest_centroids(X, initial_centroids)
print("First three elements in idx are:", idx[:3])
from public_tests import *
find_closest_centroids_test(find_closest_centroids)
# UNQ_C2
# GRADED FUNCTION: compute_centroids
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    # Useful variables
    m, n=X.shape
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    for j in range(centroids.shape[0]):
        points=X[idx==j]
        centroids[j]=np.mean(points, axis=0)
    return centroids
K=3
centroids = compute_centroids(X, idx, K)
print("The centroids are:", centroids)
compute_centroids_test(compute_centroids)
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    # Initialize values
    m, n=X.shape
    K=initial_centroids.shape[0]
    centroids=initial_centroids
    previous_centroids=centroids
    idx=np.zeros(m)
    plt.figure(figsize=(8, 6))
    # Run K-Means
    for i in range(max_iters):
        #Output progress
        print(f"K-Means iteration {i}/{max_iters-1}")
        # For each example in X, assign it to the closest centroid
        idx=find_closest_centroids(X, centroids)
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids=centroids
        # Given the memberships, compute new centroids
        centroids=compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx
# Load an example dataset
X = load_data()
# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
# Number of iterations
max_iters = 10
# Run K-Means
centroids, idx=run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    Returns:
        centroids (ndarray): Initialized centroids
    """
    # Randomly reorder the indices of examples       
    randidx=np.random.permutation(X.shape[0]) 
    # Take the first K examples as centroids
    return X[randidx[:K]]
# Run this cell repeatedly to see different outcomes.
# Set number of centroids and max number of iterations
K = 3
max_iters = 10
# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)
# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
# Load an image of a bird
original_img = plt.imread("images/bird_small.png")
plt.imshow(original_img)
plt.show()
print("Shape of original_img is:", original_img.shape)
# Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255
# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.
X_img=np.reshape(original_img, (original_img.shape[0]*original_img.shape[1], 3))
# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 50
max_iters = 20
# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K)
# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])
# Plot the colors of the image and mark the centroids
plot_kMeans_RGB(X_img, centroids, idx, K)
# Visualize the 16 colors selected
show_centroid_colors(centroids)
# Find the closest centroid of each pixel
idx=find_closest_centroids(X_img, centroids)
# Replace each pixel with the color of the closest centroid
X_recovered=centroids[idx, :]
# Reshape image into proper dimensions
X_recovered=np.reshape(X_recovered, original_img.shape)
# Display original image
fig, ax=plt.subplots(1, 2, figsize=(16, 16))
plt.axis("off")
ax[0].imshow(original_img)
ax[0].set_title("Original")
ax[0].set_axis_off()
# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title(f"Compressed with {K} colours")
ax[1].set_axis_off()
plt.show()'''
X_train, X_val, y_val = load_data()
# Display the first five elements of X_val
print("The first 5 elements of X_train are:\n", X_train[:5])  
# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5])  
print ("The shape of X_train is:", X_train.shape)
print ("The shape of X_val is:", X_val.shape)
print ("The shape of y_val is: ", y_val.shape)
# Create a scatter plot of the data. To change the markers to blue "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c="b")
# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()
# UNQ_C1
# GRADED FUNCTION: estimate_gaussian

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    Args:
        X (ndarray): (m, n) Data matrix
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n =X.shape
    #mu=np.sum(X, axis=0)/m
    #var=(np.sum((X-mu)**2, axis=0))/m
    mu=np.mean(X, axis=0)
    var=np.var(X, axis=0)
    return mu, var
# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)              
print("Mean of each feature:", mu)
print("Variance of each feature:", var)
from public_tests import *
estimate_gaussian_test(estimate_gaussian)
# Returns the density of the multivariate normal
# at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)
#Plotting code 
visualize_fit(X_train, mu, var)
plt.show()
# UNQ_C2
# GRADED FUNCTION: select_threshold
def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 
    best_epsilon=0
    best_F1=0
    F1=0
    step_size=(max(p_val)-min(p_val))/1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        tp=(p_val<epsilon)&y_val
        fp=(p_val<epsilon)&(y_val==0)
        fn=(p_val>epsilon)&y_val
        prec=np.sum(tp)/np.sum(tp+fp)
        rec=np.sum(tp)/np.sum(tp+fn)
        F1=(2*prec*rec)/(prec+rec)
        if F1>best_F1:
            best_F1=F1
            best_epsilon=epsilon
    return best_epsilon, best_F1
p_val=multivariate_gaussian(X_val, mu, var)
epsilon, F1=select_threshold(y_val, p_val)
print(f"Best epsilon found using cross-validation: {epsilon}")
print(f"Best F1 on Cross Validation Set: {F1}")
select_threshold_test(select_threshold)
# Find the outliers in the training set 
outliers = p < epsilon
# Visualize the fit
visualize_fit(X_train, mu, var)
# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro', markersize= 10,markerfacecolor='none', markeredgewidth=2)
plt.show()
# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()
print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)
# Apply the same steps to the larger dataset
# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)
# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)
print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))