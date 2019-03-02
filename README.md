# RandomizedOptimization
Project code for the machine learning course at Georgia Tech University.

Link to my code on GitHub: https://github.com/cooperdramsey/RandomizedOptimization

Running the code to repeat the analysis I did is fairly simple. I used an anaconda interpreter set to python 3.6. All of the packages loaded are specified in the requirements.txt file. You can create the exact anaconda interpreter I used by loading the pacakges found in the requirements file. The core pacakges I installed where matplotlib v3.0.2, numpy v1.15.4, pandas v0.24.0 and scikit-learn v0.20.2. All of the other packages were automatically installed with those core libraries.

All of the source code is in the main.py file. Each algorithm I tested is set up to plot learning curves based upon a set of pre-determined parameters, or based upon the results of a grid search cross validation with parameters specified in the variable param_grids. Near the top of the file right below the definition for the funtion plot_learning_curve are two variables: use_best_params and data_set. These two parameters dictate what the rest of the file does. 

**use_best_params** is a boolean. When set to True, the main.py file will use the pre-determined parameters for each algorithm thus skipping the parameter grid search cross validation. All of the results shown in my analysis used this variable set to True.
Setting the varibale to False will have each algorithm perform a grid search cross validation accross the parameters listed in the variable params_grid. NOTE: Setting this variable to False can lead to very long run times as each algorithm is trained and predicted over multiple times.

**data_set** is a string that can be set to either 'wine' or 'loan'. Setting this variable to 'wine' will have the program run each algorithm on the Wine dataset which is 1,600 rows of data. Setting this variable to 'loan' will have the program run in the Loan data set which is 30,000 rows of data.

Again, if you set **use_best_params=False** and **data_set='loan'** the runtime will be extremely long, likely a few hours.

The datasets are included in the repository and are in the root of the project so the code can read directly from them using a relative path. You don't need to change anything in the csv files for the code to work. 

Data Sets:
1. Wine Quality
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. Data available from: https://archive.ics.uci.edu/ml/datasets/wine+quality

2. UCI Credit Card Data
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. Data Available from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
