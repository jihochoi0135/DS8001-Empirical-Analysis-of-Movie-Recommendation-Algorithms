import os
import pandas as pd
import numpy as np
import time
import random

# Step 0: Same pre-procedure for every algorithms
def rmse(y_true, y_pred):
    #Convert inputs to NumPy
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    #RMSE
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def split_train_test(df, test_ratio=0.2, seed=1234):
    #Shuffle the dataset to remove ordering effects
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    #Determine the size of the test set
    n_test = int(len(df_shuffled) * test_ratio)

    #Split into test and train sets
    test = df_shuffled.iloc[:n_test].copy()
    train = df_shuffled.iloc[n_test:].copy()

    #Keep only test rows with users and movies seen in train
    mask = test["movieId"].isin(train["movieId"]) & test["userId"].isin(train["userId"])
    test = test[mask].reset_index(drop=True)
    return train, test


def evaluate_model(df, predict_fn, test_ratio=0.2, seed=1234):
    #Start timing model evaluation
    start_time = time.time()

    #Split data to train and test set
    train, test = split_train_test(df, test_ratio=test_ratio, seed=seed)

    #Take true ratings and get prediction
    y_true = test["rating"].values
    y_pred = predict_fn(train, test)

    #RMSE
    score = rmse(y_true, y_pred)

    #Take Run time
    elapsed = time.time() - start_time
    return score, elapsed, train, test


def naive_movie_average_predict(train, test):
    #find the average rating for each movie
    movie_means = train.groupby("movieId")["rating"].mean()
    #fall back value
    global_mean = movie_means.mean()
    #map movie to their mean
    preds = test["movieId"].map(movie_means).fillna(global_mean)
    return preds.values

#==small
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")    #<===== CHANGE ME
df = pd.read_csv("Master_small.csv")
seed = random.randint(1, 10_000_000)
rmse_val, time_taken,_,_ = evaluate_model(df,naive_movie_average_predict,0.2,seed)
print(f"Naive Small: RMSE={rmse_val:.4f}, time={time_taken:.4f}s")

#==large
print("loading big set")
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest")    #<===== CHANGE ME
df = pd.read_csv("Master_large.csv")
print("done loading big set")
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
        naive_movie_average_predict,
        0.2,
        seed
    )
    print(f"Naive ({n:,} rows): RMSE={rmse_val:.4f}, time={time_taken:.4f}s")
    results.append({
        "size": n,
        "rmse": rmse_val,
        "time_seconds": time_taken
    })

results_df = pd.DataFrame(results)
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project")    #<===== CHANGE ME
results_df.to_csv("naive_results.csv", index=False)
print("\nSaved results to naive_large_results.csv")


seed = random.randint(1, 10_000_000)
rmse_val, time_taken,_,_ = evaluate_model(df,naive_movie_average_predict,0.2,seed)
print(f"Naive Large: RMSE={rmse_val:.4f}, time={time_taken:.4f}s")
