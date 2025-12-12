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

# same function
def build_index_mappings(train):
    unique_users = train["userId"].unique()
    unique_movies = train["movieId"].unique()
    user_id_to_index = {uid: i for i, uid in enumerate(unique_users)}
    movie_id_to_index = {mid: i for i, mid in enumerate(unique_movies)}
    return user_id_to_index, movie_id_to_index

# same function
def train_mf_bias_model(train, n_factors, n_epochs, lr, reg):
    user_id_to_index, movie_id_to_index = build_index_mappings(train)
    n_users = len(user_id_to_index)
    n_movies = len(movie_id_to_index)

    mu = train["rating"].mean()
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    P = 0.01 * np.random.randn(n_users, n_factors)
    Q = 0.01 * np.random.randn(n_movies, n_factors)

    user_index = train["userId"].map(user_id_to_index).values
    movie_index = train["movieId"].map(movie_id_to_index).values
    ratings = train["rating"].values

    for haha in range(n_epochs):
        perm = np.random.permutation(len(train))
        for idx in perm:
            u = user_index[idx]
            i = movie_index[idx]
            r_ui = ratings[idx]

            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            e_ui = r_ui - pred

            bu[u] += lr * (e_ui - reg * bu[u])
            bi[i] += lr * (e_ui - reg * bi[i])

            p_u = P[u].copy()
            q_i = Q[i].copy()

            P[u] += lr * (e_ui * q_i - reg * p_u)
            Q[i] += lr * (e_ui * p_u - reg * q_i)
        print(f"Epoch {haha+1}/{n_epochs} done")

    model = {"mu": mu,"bu": bu,"bi": bi,
             "P": P,"Q": Q,
             "user_id_to_index": user_id_to_index,
             "movie_id_to_index": movie_id_to_index,}

    return model

# same function
def mf_bias_predict(train, test, n_factors=30, n_epochs=15, lr=0.02, reg=0.02):
    model = train_mf_bias_model(train,n_factors,n_epochs,lr,reg)
    mu = model["mu"]
    bu = model["bu"]
    bi = model["bi"]
    P = model["P"]
    Q = model["Q"]
    user_id_to_index = model["user_id_to_index"]
    movie_id_to_index = model["movie_id_to_index"]

    preds = []
    for _, row in test.iterrows():
        uid = row["userId"]
        mid = row["movieId"]
        if uid in user_id_to_index and mid in movie_id_to_index:
            u = user_id_to_index[uid]
            i = movie_id_to_index[mid]
            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
        else:
            pred = mu
        pred = np.clip(pred, 0.5, 5.0)
        preds.append(pred)

    return np.array(preds)

from itertools import product

# the tunnner, also known as the wrapper function, that is all it does, pass the grid value to mf_bias_predict
def tune_mf_hyperparams(df,param_grid,test_ratio=0.2,seed=1234):
    results = []

    for n_factors, n_epochs, lr, reg in product(
        param_grid["n_factors"],
        param_grid["n_epochs"],
        param_grid["lr"],
        param_grid["reg"]
    ):
        def predict_fn(train, test,nf=n_factors, ne=n_epochs, lr_=lr, reg_=reg):
            return mf_bias_predict(train,test,n_factors=nf,n_epochs=ne,lr=lr_,reg=reg_)

        rmse_score, elapsed, _, _ = evaluate_model(df,predict_fn=predict_fn,test_ratio=test_ratio,seed=seed)

        print(f"n_factors={n_factors:2d}, "
              f"n_epochs={n_epochs:2d}, "
              f"lr={lr:.4f}, reg={reg:.4f} "
              f"=> RMSE={rmse_score:.4f}, time={elapsed:.2f}s"
        )

        results.append({"n_factors": n_factors,
                        "n_epochs": n_epochs,
                        "lr": lr,
                        "reg": reg,
                        "rmse": rmse_score,
                        "time_sec": elapsed,})

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    # Best row (lowest RMSE)
    best_row = results_df.iloc[0]
    best_params = {
        "n_factors": int(best_row["n_factors"]),
        "n_epochs": int(best_row["n_epochs"]),
        "lr": float(best_row["lr"]),
        "reg": float(best_row["reg"]),
    }

    print("\nBest hyperparameters:")
    print(best_params)
    print(f"Best RMSE: {best_row['rmse']:.4f}")

    return results_df, best_params


if __name__ == "__main__":
    # this is the tuning grid for 100,000 rating, that is Master_small.csv
    #    Do not run this on Master_large.csv
    #        It will take more than 24 h to run
    param_grid = {
        "n_factors": [20,30,40],
        "n_epochs": [10,15],
        "lr": [0.005, 0.01,0.02],
        "reg": [0.01, 0.02, 0.05, 0.08],
    }

    os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")  # <===== CHANGE ME
    df = pd.read_csv("Master_small.csv")

    results_df, best_params = tune_mf_hyperparams(df, param_grid)

    print("\nAll results (top 5):")
    print(results_df.head())