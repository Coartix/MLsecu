_Contributors:_  
Hugo Deplagne  
Pierre Litoux  
Param Dave  
Victor Miara  
Amine El-Maghraoui  


## Outlier detection

In `OutlierIdentification.ipynb` we preprocessed the SWaT.A3 excel to a one_hot_enc dataframe that we saved in 'data/'.  
Then, used the `IsolationForest` and `LocalOutlierFactor` algorithms to detect outliers in the dataset.  

Since performance was not convincing enough, we tried to implement an LSTM trained on the normal data to detect outliers when comparing the prediction to the attacked data.  
This did not give probant results either.  

## Attack classification

In `Classification.ipynb` we used the one_hot_enc dataframe to train a `RandomForestClassifier`, a `XGboostClassifier`, a `DecisionTreeClassifier` and a `MLPClassifier`. Those, gave extremely good results on test set.  

After that we analyzed the most important columns from XGboost.
