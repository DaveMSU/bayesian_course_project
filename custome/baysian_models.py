import pymc3 as pm
import numpy as np


class BLinearRegression:
    def __init__(
            self, 
            draws: int = 2000, 
            tune: int = 1000, 
            chains: int = 1, 
            cores: int = 2, 
            add_bias_column: bool = True
        ):
        """
        TODO: write func description.
        """
        self._add_bias_column = add_bias_column
        self._pm_sample_params = dict(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            cores=cores
        )
        self._trace = None
        self.coef_ = None
        self._pm_model = None        
        
        
    def _get_correct_X(self, X: np.array) -> np.array:
        """
        TODO: write func description.
        """
        if self._add_bias_column:
            return np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        else:
            return X        
    
    
    def fit(self, X, y):
        """
        TODO: write func description.
        """
        X_ = self._get_correct_X(X)
        dim = X_.shape[1]

        with pm.Model() as linreg_model:
            X_data = pm.Data("X_data", X_)
            w = pm.Normal('w', mu=0, sd=10, shape=(dim))
            sigma = pm.HalfNormal('sigma', sd=10)
            outputs = pm.Normal(
                'y', 
                mu=pm.math.dot(w, X_.T), 
                sd=sigma,
                observed=y
            )
            self._trace = pm.sample(**self._pm_sample_params)
        self.coef_ = np.mean(self._trace.get_values('w'), axis=0)
        self._pm_model = linreg_model

    
    def predict(self, X: np.array) -> np.array:
        """
        TODO: write fun
        """
        X_ = self._get_correct_X(X)        
        y_pred = X_.dot(self.coef_).reshape(-1)
        return y_pred
    
    
    def predict_confidence(self, X: np.array) -> np.array:
        """
        TODO: write func description.
        """
        X_ = self._get_correct_X(X)
        all_w = self.trace.get_values('w')
        y_sd_pred = X_.dot(all_w.T).std(axis=1)
        return y_sd_pred


    @property
    def trace(self) -> pm.backends.base.MultiTrace:
        return self._trace
        