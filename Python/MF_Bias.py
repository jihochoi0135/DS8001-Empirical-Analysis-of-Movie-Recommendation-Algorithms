import os
import pandas as pd
import numpy as np
import time
import random


# Step 0: Same pre-procedure for every algorithms
def rmse(y_true, y_pred):
    # Convert inputs to NumPy
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # RMSE
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def split_train_test(df, test_ratio=0.2, seed=1234):
    # Shuffle the dataset to remove ordering effects
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Determine the size of the test set
    n_test = int(len(df_shuffled) * test_ratio)

    # Split into test and train sets
    test = df_shuffled.iloc[:n_test].copy()
    train = df_shuffled.iloc[n_test:].copy()

    # Keep only test rows with users and movies seen in train
    mask = test["movieId"].isin(train["movieId"]) & test["userId"].isin(train["userId"])
    test = test[mask].reset_index(drop=True)
    return train, test


def evaluate_model(df, predict_fn, test_ratio=0.2, seed=1234):
    # Start timing model evaluation
    start_time = time.time()

    # Split data to train and test set
    train, test = split_train_test(df, test_ratio=test_ratio, seed=seed)

    # Take true ratings and get prediction
    y_true = test["rating"].values
    y_pred = predict_fn(train, test)

    # RMSE
    score = rmse(y_true, y_pred)

    # Take Run time
    elapsed = time.time() - start_time
    return score, elapsed, train, test

def build_index_mappings(train):
    #take all unique movie and user id
    unique_users = train["userId"].unique()
    unique_movies = train["movieId"].unique()
    #map it to their consecutive indices
    user_id_to_index = {uid: i for i, uid in enumerate(unique_users)}
    movie_id_to_index = {mid: i for i, mid in enumerate(unique_movies)}
    return user_id_to_index, movie_id_to_index

# where training happens
def train_mf_bias_model(train, n_factors, n_epochs, lr, reg):
    #build index mappings for users and movies
    user_id_to_index, movie_id_to_index = build_index_mappings(train)
    n_users = len(user_id_to_index)
    n_movies = len(movie_id_to_index)

    #init value
    mu = train["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    P = 0.01 * np.random.randn(n_users, n_factors)
    Q = 0.01 * np.random.randn(n_movies, n_factors)

    #convert user and movie IDs to index form
    user_index = train["userId"].map(user_id_to_index).values
    movie_index = train["movieId"].map(movie_id_to_index).values
    ratings = train["rating"].values

    #train model using gradient descent with learning rate and regularization
    for haha in range(n_epochs):
        perm = np.random.permutation(len(train))
        for idx in perm:
            u = user_index[idx]
            i = movie_index[idx]
            r_ui = ratings[idx]

            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            e_ui = r_ui - pred

            #update
            bu[u] += lr * (e_ui - reg * bu[u])
            bi[i] += lr * (e_ui - reg * bi[i])
            p_u = P[u].copy()
            q_i = Q[i].copy()
            P[u] += lr * (e_ui * q_i - reg * p_u)
            Q[i] += lr * (e_ui * p_u - reg * q_i)
        #print(f"Epoch {haha+1}/{n_epochs} done")      #<===== uncommon to see more detail

    model = {"mu": mu,"bu": bu,"bi": bi,
             "P": P,"Q": Q,
             "user_id_to_index": user_id_to_index,
             "movie_id_to_index": movie_id_to_index,}

    return model

# note in this function the 4 parameter defult value is already tuned in MF Tuning
#       the only difference is in MF tuning there is one more wrapper that pass the
#       different parameter values
def mf_bias_predict(train, test, n_factors=30, n_epochs=15, lr=0.02, reg=0.02):
    # this is a wrapper
    # Extract trained model parameters
    model = train_mf_bias_model(train,n_factors,n_epochs,lr,reg)
    mu = model["mu"]
    bu = model["bu"]
    bi = model["bi"]
    P = model["P"]
    Q = model["Q"]
    user_id_to_index = model["user_id_to_index"]
    movie_id_to_index = model["movie_id_to_index"]

    #generate predictions
    preds = []
    for _, row in test.iterrows():
        uid = row["userId"]
        mid = row["movieId"]
        if uid in user_id_to_index and mid in movie_id_to_index:
            u = user_id_to_index[uid]
            i = movie_id_to_index[mid]
            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
        else:
            #fall back val
            pred = mu
        pred = np.clip(pred, 0.5, 5.0)
        preds.append(pred)

    return np.array(preds)


os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")    #<===== CHANGE ME
df = pd.read_csv("Master_small.csv")
seed = random.randint(1, 10_000_000)
rmse_val, time_taken, _, _ = evaluate_model(df, mf_bias_predict, 0.2, seed)
print(f"MF+Bias : RMSE={rmse_val:.4f}, time={time_taken:.4f}s")

#==large
print("laoding big set")
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest")    #<===== CHANGE ME
df = pd.read_csv("Master_large.csv")
print("done laoding big set")


sizes = [100000, 390000, 680000, 970000, 1260000,
    1550000, 1840000, 2130000, 2420000, 2710000,
    3000000]
results = []
for n in sizes:
    print(f"\n=== Running on first {n:,} rows ===")
    df_sub = df.iloc[:n].copy()
    seed = random.randint(1, 10_000_000)
    rmse_val, time_taken, _, _ = evaluate_model(
        df_sub,
        mf_bias_predict,
        0.2,
        seed
    )
    print(f"MF+Bias ({n:,} rows): RMSE={rmse_val:.4f}, time={time_taken:.4f}s")
    results.append({
        "size": n,
        "rmse": rmse_val,
        "time_seconds": time_taken
    })

results_df = pd.DataFrame(results)
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project")    #<===== CHANGE ME
results_df.to_csv("MF_Bias.csv", index=False)
print("\nSaved results to MF_Bias.csv")

###########################################################
#######  This code will run for 1.5 h #####################
#####                                                  ####
# uncommon it if you have time, but idk if you have time for it to run,
#    result is in the pdf, look for the 33,000,000 row, it is all recored

#seed = random.randint(1, 10_000_000)
#rmse_val, time_taken,_,_ = evaluate_model(df,mf_bias_predict,0.2,seed)
#print(f"MF+Bias: RMSE={rmse_val:.4f}, time={time_taken:.4f}s")

###########################################################