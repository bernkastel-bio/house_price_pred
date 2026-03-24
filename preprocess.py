import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union



def get_safe_n_splits(y, requested_splits=10):
    """Return the maximum n_splits that won't cause stratification issues"""
    unique_counts = pd.Series(y).value_counts()
    min_count = unique_counts.min()
    return min(requested_splits, max(2, min_count))


def target_enc_bayes_smoothing(
        x: np.ndarray, 
        y: np.ndarray,
        n_kf_splits: int,
        smoothing_factor: float, 
        noise_std: int = 0
        ) -> np.ndarray:
    """
    Applies target encoding with bayesian smoothign to selected discrete variable.

    Parameters:
    -----------
    x : np.ndarray
        Categorical variable
    y : np.ndarray
        Target variable
    n_kf_splits : int
        Number of splits for KFold cross-validation
    smoothing_factor :

    """
    kf = KFold(n_splits=n_kf_splits, shuffle=True, random_state=0xC0FFEE)
    encoded_values = np.zeros(x.shape[0])

    glob_mean = y.mean()

    for j, (train_idx, val_idx) in enumerate(kf.split(x)):
        train_stats = {}
        
        for cat in np.unique(x[train_idx]):
            mask = x[train_idx] == cat
            cat_mean = y[train_idx][mask].mean()
            cat_count = mask.sum()

            smoothed_mean = (cat_count * cat_mean + smoothing_factor * glob_mean) / (cat_count + smoothing_factor)

            confidence = cat_count / (cat_count + smoothing_factor)

            train_stats[cat] = {
                'enc_value': smoothed_mean,
                'conf': confidence,
                'sample_size': cat_count, 
                'raw_mean': cat_mean
            }

        for i, cat in enumerate(x[val_idx]):
            enc_val = train_stats.get(cat, {}).get('enc_value', glob_mean)

            if noise_std > 0:
                
                enc_val += np.random.normal(0, noise_std)
            
            encoded_values[val_idx[i]] = enc_val
        
    return encoded_values, train_stats


def regression_imputer(
        data: pd.DataFrame,
        x_features: List[str],
        target: str,
        model: Optional[Any] = None
        ):
    """
    Method for feature imputing
    """
    if model is None:
        model = BayesianRidge()

    mask = data[target].isnull()

    train = data.iloc[~mask]
    test = data.iloc[mask]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0xC0FFEE)

    model.fit(train[x_features].to_numpy(), train[target].values)

    scores = cross_val_score(
        estimator=model, 
        X=train[x_features].to_numpy(), 
        y=train[target].values, 
        cv=kf,
        n_jobs=-1)
    for i, score in enumerate(scores):
        print(f"Fold {i}: {score}")
    
    imputed_target = model.predict(test[x_features].to_numpy())

    test[target] = imputed_target

    result = pd.concat([train, test]).sort_values(by='Id')
    
    return result


# def timedata_enc(
#         data: pd.DataFrame,
#         col: str):
#     normalized_x = (data[col].values + 2*np.pi / 2) % 2*np.pi - 2*np.pi / 2
    

