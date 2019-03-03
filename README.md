# RandomizedOptimization
Project code for the machine learning course at Georgia Tech University.

Link to my code on GitHub: https://github.com/cooperdramsey/RandomizedOptimization

Running the code to repeat the analysis I did is fairly simple. I used an anaconda interpreter set to python 3.6. All of the packages loaded are specified in the requirements.txt file. You can create the exact anaconda interpreter I used by loading the pacakges found in the requirements file. The core pacakges I installed where matplotlib v3.0.2, numpy v1.15.4, pandas v0.24.0, scikit-learn v0.20.2, and mlrose v1.0.1. Note that
I ended up using a direct download of mlrose from the github page as there were updates that had not yet been uploaded to the pip installer.
The Github link is: https://github.com/gkhayes/mlrose

All of the source code for the neural network portion can be found in nn.py. THe code for the optimization problem section can be found in opt.py.

Running the file is fairly straight forward, but to view specific analysis you will need to comment out the sections you don't want see. Each file has each analysis from top to bottom in executed one after the other so commenting out code is easy and won't break the analysis you want to run.

All of the graphs used for the analysis are included in the Graphs directory as well as a few extra not included in the analysis report.

The dataset is included in the repository and is in the root of the project so the code can read directly from them using a relative path. You don't need to change anything in the csv files for the code to work.

Data Set:
1. UCI Credit Card Data
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. Data Available from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients