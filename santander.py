import pandas as pd
import numpy as np
import dask.dataframe
import sys
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from joblib import parallel_backend
from sklearn.model_selection import cross_val_score


def main():
    print("--- Reading train ---")
    train = pd.read_csv('train.csv')

    # removing constant columns
    y = train['target'].to_numpy()
    y = y.reshape((len(y),-1))

    train.drop(['ID','target'], axis=1, inplace=True)
    feature_selector = VarianceThreshold()
    feature_selector.fit(train)
    train = feature_selector.transform(train)
 
    # log transformation
    train = np.hstack((y,train))
    train = np.log1p(train)
    
    # train-dev splitting
    print("original train size: ", train.shape)

    X_train = train[:, 1:]
    y_train = train[:, 0]

    # dimensionality reduction
    pca = PCA(n_components=0.8, svd_solver='full')
    # If 0 < n_components < 1 --> select the number of components st amount of variance that needs to be explained
    #  is greater than the percentage specified by n_components.
    pca.fit(X_train)
    Xred_train = pca.transform(X_train)
    print("reduced train size: ", Xred_train.shape)


    # naive method
    print("===== Naive model (mean) ======")
    from sklearn.dummy import DummyRegressor
    regDummy = DummyRegressor(strategy="mean")
    regDummy.fit(Xred_train, y_train)
    y_train_pred = regDummy.predict(Xred_train)
    print("training accuracy = %.3f" %mean_squared_error(y_train.T, y_train_pred))

    # Lasso method
    print("===== Lasso regression model =====")
    from sklearn import linear_model
    regLasso = linear_model.Lasso()
    scores = cross_val_score(regLasso, Xred_train, y_train, scoring='neg_mean_squared_error', cv=8, n_jobs=-1) 
    print("training accuracy = %.3f" % -np.mean(scores))

    # Random forest method
    print("===== Random forest model =====")
    from sklearn.ensemble import RandomForestRegressor
    regRF = RandomForestRegressor()
    scores = cross_val_score(regRF, Xred_train, y_train, scoring='neg_mean_squared_error', cv=8, n_jobs=-1) 
    print("training accuracy = %.3f" % -np.mean(scores))

    # gradient boosting
    print("===== Gradient boosting method =====")
    from sklearn.ensemble import GradientBoostingRegressor
    regBoost = GradientBoostingRegressor()
    scores = cross_val_score(regBoost, Xred_train, y_train, scoring='neg_mean_squared_error', cv=8, n_jobs=-1) 
    print("training accuracy = %.3f" % -np.mean(scores))

    # lightGBM
    print("===== lightGBM =====")
    import lightgbm as lgb
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1','l2'],
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 8,
        "num_leaves": 128,  
        "max_bin": 512,
        "num_iterations": 100000
    }
    regGBM = lgb.LGBMRegressor()
    scores = cross_val_score(regGBM, Xred_train, y_train, scoring='neg_mean_squared_error', cv=8, n_jobs=-1) 
    print("training accuracy = %.3f" % -np.mean(scores))

    # voting
    print("===== Voting method =====")
    from sklearn.ensemble import VotingRegressor
    regVoting = VotingRegressor([('rf',regRF), ('boost',regBoost), ('lgbm',regGBM)], n_jobs=-1)
    regVoting.fit(Xred_train, y_train)
    y_train_pred = regVoting.predict(Xred_train)
    print("training accuracy = %.3f" %mean_squared_error(y_train.T, y_train_pred))


    print("--- Reading test ---")
    chunk = pd.read_csv('test.csv', chunksize = 4000)
    test = pd.concat(chunk)
    # test = dask.dataframe.read_csv('test.csv')

    ids = test['ID'].to_numpy()
    test.drop(['ID'], axis=1, inplace=True)
    test = feature_selector.transform(test)
    test = np.log1p(test)
    
    regressor = regVoting
    Xred_test = pca.transform(test)
    predictions = np.expm1(regressor.predict(Xred_test))
    
    predictions = predictions.reshape((len(predictions),-1))
    ids = ids.reshape((len(ids),-1))
    results = pd.DataFrame(data=np.hstack((ids,predictions)), columns=["ID", "target"])
    results.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()





