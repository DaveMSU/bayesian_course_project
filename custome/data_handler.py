import typing as tp
import pandas as pd


class DataFrameHandler:
    def __init__(self):
        pass
    
    def _check_columns(self, df: pd.DataFrame, cols: tp.List[str]) -> bool:
        """
        TODO: write func description.
        """
        columns_set = set(cols)
        df_columns_set = set(df.columns)
        intersection_of_them = columns_set.intersection(df_columns_set)
        return len(intersection_of_them) == len(columns_set)
    
    def prepare_dataset(
            self,
            dataset: pd.DataFrame, 
            to_drop_columns: tp.List[str],
            columns_to_existance_checking: tp.List[str]
        ) -> pd.DataFrame:
        """
        TODO: write func description.
        """
        df = dataset.copy()
        df = pd.concat((df, pd.get_dummies(df.sex)), axis=1)
        df = pd.concat((df, pd.get_dummies(df.region)), axis=1)
        df = df.drop(to_drop_columns, axis=1)
        assert self._check_columns(df, columns_to_existance_checking)
        return df
    
    def count_statistics(
            self,
            dataset: pd.DataFrame,
            columns: tp.List[str]
        ) -> tp.Dict[str, tp.Dict[str, float]]:
        """
        TODO: write func description.
        """
        df = dataset
        statictics = dict()
        assert self._check_columns(df, columns)
        for column in columns:
            statictics[column] = dict()
            statictics[column]['mean'] = df[column].mean()
            statictics[column]['std'] = df[column].std()
        return statictics
    
    def normalize_dataset(
            self,
            dataset: pd.DataFrame,
            norm_dict: tp.Dict[str, tp.Dict[str, float]],
        ) -> pd.DataFrame:
        """
        TODO: write func description.
        """
        df = dataset.copy()
        for column, norm_params in norm_dict.items():
            mean_ = norm_params['mean']
            std_ = norm_params['std']
            df[column] = (df[column] -  mean_) / std_
        return df
    