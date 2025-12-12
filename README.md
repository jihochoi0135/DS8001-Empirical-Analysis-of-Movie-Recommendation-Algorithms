# DS8001-Empirical-Analysis-of-Movie-Recommendation-Algorithms

1. Download each .py file from the "Python" folder

2. Download both datasets, Master_small.csv and Master_large.csv
  You will only see Master_small.csv.  Master_large is too big, it is 3.1G.  Go here https://drive.google.com/drive/folders/1q5xe-ZbaV3J3cjeugr1TzCEd4jcTWpT9?usp=sharing
    or generate using Pre_process.py, following the instructions below

5. If you are running from the raw data from movielens
<br>You will run Pre_process.py twice.
<br>First Run: change file path to where ml-latest-small is, then change output file name to Master_small.csv
<br>Second Run: change file path to where ml-latest is, then change output file name to Master_large.csv  <- everything involving this file takes a long time

6. Edit the path of the dataset in each .py file:
<br>a. Go to the bottom of the file.
<br>b. Replace the file path to where "Master_small.csv" and "Master_large.csv" are located.


7. After completing editing, go to the command prompt and run the following command.
<br>a. python ___.py
<br>b. replace ___ with the file name.
<br>c. e.g. python user_based.py
<br>d. Each file, except Tuning and Smooth, will generate a CSV file containing runtime and RMSE.
<br>e. Loading Master_large.csv take a long time.

Smooth.py will perform tuning as it runs, and all the results will be printed on screen.  At the end, press Enter to exit the code
