import os
import pandas as pd
import numpy as np
import time
import random

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def split_train_test(df, test_ratio=0.2, seed=42):
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(df_shuffled) * test_ratio)

    test = df_shuffled.iloc[:n_test].copy()
    train = df_shuffled.iloc[n_test:].copy()

    # (optional) keep only users & movies that appear in train
    mask = test["movieId"].isin(train["movieId"]) & test["userId"].isin(train["userId"])
    test = test[mask].reset_index(drop=True)

    return train, test

def evaluate_model(df, predict_fn, test_ratio=0.2, seed=42):
    start_time = time.time()

    train, test = split_train_test(df, test_ratio=test_ratio, seed=seed)

    y_true = test["rating"].values
    y_pred = predict_fn(train, test)

    score = rmse(y_true, y_pred)
    elapsed = time.time() - start_time

    return score, elapsed, train, test

def bias_baseline_predict(train, test):
    #-- mu
    mu = train["rating"].mean()

    #-- b_i
    movie_mean = train.groupby("movieId")["rating"].mean()
    movie_bias = movie_mean - mu  # b_i

    #-- b_u
    user = train.copy()
    user["movie_bias"] = user["movieId"].map(movie_bias).fillna(0.0)
    user["residual"] = user["rating"] - mu - user["movie_bias"]
    user_bias = user.groupby("userId")["residual"].mean()

    mb = test["movieId"].map(movie_bias).fillna(0.0)  # <--- movie bias
    ub = test["userId"].map(user_bias).fillna(0.0)    # <--- user bias

    preds = mu + mb + ub
    preds = np.clip(preds, 0.5, 5.0)
    return preds.values

os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")
df = pd.read_csv("Master_small.csv")
seed = random.randint(1, 10_000_000)
rmse_val, time_taken, _, _ = evaluate_model(df, bias_baseline_predict, 0.2, seed)
print(f"Bias Based : RMSE={rmse_val:.4f}, time={time_taken:.4f}s")

#==large
print("loading big set")
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest")
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
        bias_baseline_predict,
        0.2,
        seed
    )
    print(f"Bias ({n:,} rows): RMSE={rmse_val:.4f}, time={time_taken:.4f}s")
    results.append({
        "size": n,
        "rmse": rmse_val,
        "time_seconds": time_taken
    })

results_df = pd.DataFrame(results)
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project")
results_df.to_csv("bias_results.csv", index=False)
print("\nSaved results to bias_large_results.csv")

#== large
seed = random.randint(1, 10_000_000)
rmse_val, time_taken,_,_ = evaluate_model(df,bias_baseline_predict,0.2,seed)
print(f"Bias Large: RMSE={rmse_val:.4f}, time={time_taken:.4f}s")

