# DS8001-Empirical-Analysis-of-Movie-Recommendation-Algorithms

1. Download each .py file from the "Python" folder

2. Download both datasets, Master_small.csv and Master_large.csv

3. If you are running from the raw data from movielens
<br>You will run Pre_process.py twice.
<br>First Run: change file path to where ml-latest-small is, then change output file name to Master_small.csv
<br>Second Run: change file path to where ml-latest is, then change output file name to Master_large.csv

4. Edit the path of the dataset in each .py file:
<br>a. Go to the bottom of the file.
<br>b. Replace the file path to where "Master_small.csv" and "Master_large.csv" are located.


5. After completing editing, go to the command prompt and run the following command.
<br>a. python ___.py
<br>b. replace ___ with the file name.
<br>c. e.g. python user_based.py
<br>d. Each file, except Tuning and Preprocess, will generate a CSV file containing runtime and RMSE.
<br>e. Loading Master_large.csv take a long time.
