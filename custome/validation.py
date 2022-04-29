import typing as tp
import numpy as np
import pandas as pd


class CrossValSampler:
    def __init__(
            self, 
            X: np.array, 
            y: np.array, 
            val_size: int = 1, 
            permutate: tp.Optional[int] = None
        ):
        self._cur_fold_num = 0
        self._X = X
        self._y = y
        self._data_len = len(X)
        self._folds_num = len(X) // val_size
        self._val_size = val_size
        
        if permutate:
            indexes = np.permutation(np.arrange(self._data_len), seed=permutate)
            self._X = self._X[indexes]
            self._y = self._y[indexes]
            
    
    @property
    def folds_num(self):
        return self._folds_num
        
        
    def __iter__(self):
        self._cur_fold_num = 0
        return self
    
    
    def __next__(self):
        if self._folds_num <= self._cur_fold_num:
            raise StopIteration()
        val_section_start_index = self._cur_fold_num * self._val_size
        val_section_end_index = val_section_start_index + self._val_size
        
        slice_before_val = list(range(0, val_section_start_index))
        val_slice = list(range(val_section_start_index, val_section_end_index))
        slice_after_val = list(range(val_section_end_index, self._data_len))
        train_slice = slice_before_val + slice_after_val

        train = [
            self._X[train_slice], 
            self._y[train_slice]
        ]
        val = [
            self._X[val_slice], 
            self._y[val_slice]
        ]
        
        self._cur_fold_num += 1
        return train, val
            
        
    def __call__(self):
        return next(self)

    
    
def get_Xy(
        df: pd.DataFrame, 
        target_name: str
    ) -> tp.Tuple[np.array, np.array]:
    """
    TODO: write func description.
    """
    X = df.drop(target_name, axis=1).values
    y = df[target_name].values
    return X, y