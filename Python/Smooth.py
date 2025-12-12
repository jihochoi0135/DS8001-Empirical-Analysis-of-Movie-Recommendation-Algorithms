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


def smoothed_movie_average_predict(train, test, k=2):
    #get global mean of all movie rating
    global_mean = train["rating"].mean()
    #find the mean and num of rating for each movie
    movie_stats = (train.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "movie_mean", "count": "movie_count"})
    )

    # the smoothing function shown in pdf
    movie_stats["smoothed_mean"] = ((movie_stats["movie_mean"] * movie_stats["movie_count"] + k * global_mean)/
                                    (movie_stats["movie_count"] + k))
    # map to the movie
    preds = test["movieId"].map(movie_stats["smoothed_mean"])
    preds = preds.fillna(global_mean)
    return preds.values


def tune_weighted_movie_average(df, k_values, test_ratio=0.2, seed=1234):
    results = []
    for k in k_values:

        def predict_fn(train, test, k_in=k):
            return smoothed_movie_average_predict(train, test, k=k_in)

        rmse, elapsed,_,_ = evaluate_model(df,predict_fn,test_ratio,seed)
        #print(f"k={k:2d} : RMSE={rmse:.4f}, time={elapsed:.4f}s")    #<===== uncommon to see more detail
        results.append({"k": k,"rmse": rmse,"time_sec": elapsed})

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    best_k = int(results_df.iloc[0]["k"])

    #print("\nBest k found:")
    print(f"k = {best_k}, RMSE = {results_df.iloc[0]['rmse']:.4f}, TIME = {results_df.iloc[0]['time_sec']:.4f}")

    return results_df, best_k

#k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30]
#results_df, best_k = tune_weighted_movie_average(df, k_values)

os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")    #<===== CHANGE ME
df = pd.read_csv("Master_small.csv")
seed = random.randint(1, 10_000_000)
k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30]
results_df, best_k = tune_weighted_movie_average(df, k_values,seed=seed)

print("loading big set")
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest")    #<===== CHANGE ME
df = pd.read_csv("Master_large.csv")
print("done loading big set")

results_df, best_k = tune_weighted_movie_average(df, k_values,seed=seed)
sizes = [100000, 390000, 680000, 970000, 1260000,
    1550000, 1840000, 2130000, 2420000, 2710000,
    3000000]
results = []
for n in sizes:
    print(f"\n=== Running on first {n:,} rows ===")
    df_sub = df.iloc[:n].copy()
    seed = random.randint(1, 10_000_000)
    results_df, best_k = tune_weighted_movie_average(df_sub, k_values,seed=seed)

#==large
seed = random.randint(1, 10_000_000)
k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30]

exit_the_program = input("ENTER to exite the program")

