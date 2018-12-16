from enum import Enum
from typing import Dict
import pandas as pd
import numpy as np
from pandas import DataFrame

class DataType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1

class MICE:
    def __init__(self,n_iter,regressor,classifier,imputation_method):
        self.n_iter = n_iter
        self.regressor = regressor
        self.classifier = classifier
        if type(imputation_method) == dict:
            assert DataType.DISCRETE in imputation_method, "No Imputation Method For Discrete Variables"
            assert DataType.CONTINUOUS in imputation_method, "No Imputation Method For Continuous Variables"
            self.imputation_method = imputation_method
        else:
            self.imputation_method = {
                DataType.DISCRETE : imputation_method,
                DataType.CONTINUOUS : imputation_method
            }

    def fit(self,X:DataFrame,data_types = None):
        # aliases
        DTC = DataType.CONTINUOUS
        #Check that data_types is valid
        if data_types == None:
            data_types = {}
            for col in X.columns: # initialize data type for each column as CONTINOUS
                data_types[col] = DTC
        if type(data_types) == dict:
            for col in X.columns:
                assert col in data_types, "Data Type not given for {}".format(col)
                assert type(data_types[col]) == DataType, "Invalid Data Type given for {}".format(col)
        else:
            assert len(data_types) == X.columns.size, "Provided Data types not equal to number of columns"
            for data_type in data_types:
                assert type(data_type)==DataType, "Invalid data type"
            #convert data_type into dictionary
            _data_types = data_types
            data_types = {}
            for i,col in enumerate(X.columns):
                data_types[col] = _data_types[i]


        missing_cols = X.columns[X.apply(lambda x:x.isna().values.any())].values
        observed = X.applymap(lambda x: not pd.isna(x))
        X = X.copy()

        for col in missing_cols:
            observed_values = X[~X[col].isna()][col].values.astype(np.float)
            imp_value = self.imputation_method[data_types[col]](observed_values)
            X[col] = X[col].map(lambda x: imp_value if pd.isna(x) else x)

        for _ in range(self.n_iter):
            for col in missing_cols:
                X_ind = X.drop([col],axis=1).values
                X_obs = X[~X[col].isna()]
                X_obs_ind = X_obs.drop([col],axis=1).values
                X_obs_dep = X_obs[col].values
                model = self.regressor if data_types[col] == DTC else self.classifier
                assert model is not None, "{} not provided.".format("Regressor" if data_types[col]==DTC else "Classifier")
                model.fit(X_obs_ind,X_obs_dep)
                l = X[col].size
                for i in range(l):
                    if not observed.loc[i,col]:
                        X.loc[i,col] = model.predict([X_ind[i]])[0]

        return X