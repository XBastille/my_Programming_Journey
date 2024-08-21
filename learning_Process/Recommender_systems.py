import itertools
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
import keras
import tabulate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import disable_eager_execution
from recsysNN_utils import *
from recsys_utils import *
from public_tests import *
#Load data
'''X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()
print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)
#  From the matrix, we can compute statistics like average rating.
tsmean=np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )
# GRADED FUNCTION: cofi_cost_func
# UNQ_C1
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu=Y.shape
    J=sum(
        np.square(R[i, j]*(np.dot(W[j, :], X[i, :])+b[0, j]-Y[i, j]))
        for j, i in itertools.product(range(nu), range(nm))
    )
    J/=2
    J+=((lambda_/2)*(np.sum(np.square(W))+np.sum(np.square(X))))
    return J
# Reduce the data set size so that this runs faster
num_users_r=4
num_movies_r=5 
num_features_r=3
X_r=X[:num_movies_r, :num_features_r]
W_r=W[:num_users_r,  :num_features_r]
b_r=b[0, :num_users_r].reshape(1,-1)
Y_r=Y[:num_movies_r, :num_users_r]
R_r=R[:num_movies_r, :num_users_r]
# Evaluate cost function
J=cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")
# Evaluate cost function with regularization 
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")
# Public tests
from public_tests import *
test_cofi_cost_func(cofi_cost_func)
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j=(tf.linalg.matmul(X, tf.transpose(W))+b-Y)*R
    return 0.5*tf.reduce_sum(j**2)+(lambda_/2)*(
        tf.reduce_sum(W**2)+tf.reduce_sum(X**2)
    )
# Evaluate cost function
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")
# Evaluate cost function with regularization 
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")
movieList, movieList_df = load_Movie_List_pd()
my_ratings=np.zeros(num_movies)          #  Initialize my ratings
# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700]=5
#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609]=2;
# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]=5   # Lord of the Rings: The Return of the King, The
my_ratings[246]=5   # Shrek (2001)
my_ratings[2716]=3   # Inception
my_ratings[1150]=5   # Incredibles, The (2004)
my_ratings[382]=2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]=5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]=5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]=3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925]=1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937]=1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]=5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i]>0]
print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i]>0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');
# Reload ratings
Y, R=load_ratings_small()
# Add new user ratings to Y
Y=np.c_[my_ratings, Y]
# Add new user indicator matrix to R
R=np.c_[(my_ratings!=0).astype(int), R]
# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)
#  Useful Values
num_movies, num_users=Y.shape
num_features=100
# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234)   # for consistent results
W=tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name="W")
X=tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name="X")
b=tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name="B")
# Instantiate an optimizer.
optimizer=keras.optimizers.Adam(learning_rate=1e-1)
iterations=200
lambda_=1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:
        # Compute the cost (forward pass included in cost)
        cost_value=cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads=tape.gradient(cost_value, [X, W, b])
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, [X, W, b]))
    # Log periodically.
    if iter%20==0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
# Make a prediction using trained weights and biases
p=np.matmul(X.numpy(), np.transpose(W.numpy()))+b.numpy()
#restore the mean
pm=p+Ymean
my_predictions=pm[:, 0]
# sort predictions
ix=tf.argsort(my_predictions, direction="DESCENDING")
for i in range(17):
    j=ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')
print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i]>0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
filter=(movieList_df["number of ratings"]>20)
movieList_df["pred"]=my_predictions
movieList_df=movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
print(movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False))'''
pd.set_option("display.precision", 1)
top10_df=pd.read_csv("./data/content_top10_df.csv")
bygenre_df=pd.read_csv("./data/content_bygenre_df.csv")
print(top10_df)
print(bygenre_df)
# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre=load_data()
num_user_features=user_train.shape[1]-3  # remove userid, rating count and ave rating during training
num_item_features=item_train.shape[1]-1  # remove movie id at train time
uvs=3  # user genre vector start
ivs=3  # item genre vector start
u_s=3  # start of columns to use in training, user
i_s=1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")
print(pprint_train(user_train, user_features, uvs,  u_s, maxcount=5))
print(pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False))
print(f"y_train[:5]: {y_train[:5]}")
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train
scalerItem=StandardScaler()
scalerItem.fit(item_train)
item_train=scalerItem.transform(item_train)
scalerUser=StandardScaler()
scalerUser.fit(user_train)
user_train=scalerUser.transform(user_train)
scalerTarget=MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train=scalerTarget.transform(y_train.reshape(-1, 1))
print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
item_train, item_test=train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test=train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test=train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")
print(pprint_train(user_train, user_features, uvs, u_s, maxcount=5))
# GRADED_CELL
# UNQ_C1
num_outputs=32
tf.random.set_seed(1)
user_NN=tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_outputs, activation="linear")
    ]
)
item_NN=tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_outputs, activation="linear")
    ]
)
# create the user input and point to the base network
user_input=tf.keras.layers.Input(shape=(num_user_features, ))
vu=user_NN(user_input)
vu=tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)
# create the item input and point to the base network
item_input=tf.keras.layers.Input(shape=(num_item_features, ))
vm=item_NN(item_input)
vm=tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)
# compute the dot product of the two vectors vu and vm
output=tf.keras.layers.Dot(axes=1)([vu, vm])
# specify the inputs and output of the model
model=tf.keras.Model([user_input, item_input], output)
model.summary()
cost_fn=tf.keras.losses.MeanSquaredError()
opt=keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=opt,
    loss=cost_fn
)
model.fit(
    [
        user_train[:, u_s:],
        item_train[:, i_s:]
    ],
    y_train,
    epochs=30
)
model.evaluate(
    [
        user_test[:, u_s:],
        item_test[:, i_s:]
    ],
    y_test
)
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))
# scale our user and item vectors
suser_vecs=scalerUser.transform(user_vecs)
sitem_vecs=scalerItem.transform(item_vecs)
# make a prediction
y_p=model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])
# unscale y prediction 
y_pu=scalerTarget.inverse_transform(y_p)
# sort the results, highest prediction first
sorted_index=np.argsort(-y_pu, axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu=y_pu[sorted_index]
sorted_items=item_vecs[sorted_index]  #using unscaled vectors for display
print(print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10))
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)
# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)
# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])
# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)
# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]
#print sorted predictions for movies rated by the user
print(print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50))
# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    return np.sum(np.square(a-b))

a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])
print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")
print(f"squared distance between a2 and b2: {sq_dist(a2, b2):0.3f}")
print(f"squared distance between a3 and b3: {sq_dist(a3, b3):0.3f}")
# Public tests
test_sq_dist(sq_dist)
input_item_m=tf.keras.layers.Input(shape=(num_item_features, ))
vm_m=item_NN(input_item_m)
vm_m=tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm_m)
model_m=tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()
scaled_item_vecs=scalerItem.transform(item_vecs)
vms=model_m.predict(scaled_item_vecs[:, i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")
count=50  # number of movies to display
dim=len(vms)
dist=np.zeros((dim, dim))
for i in range(dim):
    for j in range(dim):
        dist[i, j]=sq_dist(vms[i, :], vms[j, :])
m_dist=ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal
disp=[["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx=np.argmin(m_dist[i])
    movie1_id=int(item_vecs[i, 0])
    movie2_id=int(item_vecs[min_idx, 0])
    disp.append(
        [movie_dict[movie1_id]["title"], movie_dict[movie1_id]["genres"],
        movie_dict[movie2_id]["title"], movie_dict[movie2_id]["genres"]] 
    )
table=tabulate.tabulate(disp, tablefmt="html", headers="firestrow")
print(table)