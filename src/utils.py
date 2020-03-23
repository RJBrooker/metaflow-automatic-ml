import pandas as pd , numpy as np
from tqdm import tqdm
import pandas as pd 
import feather
from sklearn.utils.class_weight import compute_class_weight
from typing import List



def infer_column_types(df : pd.DataFrame ) -> pd.DataFrame :
    """ Guesses data types columns to the appropriate types. """
    for col in tqdm(df.columns):
        col_type = df[col].dtype
        if '_id' in col:
            col_type= 'object'
        elif (col_type in ['object',np.dtype('O')]) :
            col_type = 'category'
        if 'date' in col:
            col_type = 'datetime64[ns]'
        yield col, col_type 



def random_dates(start: pd.Timestamp , end: pd.Timestamp , n:int =10) -> list :
    """ Create a list of random dates that are evenly distributes between two dates"""
    start_u = start.value//10**9
    end_u = end.value//10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')



def test_train_temporal_split(X :pd.DataFrame , y:pd.Series , dates: pd.Series ) -> tuple :
    """ Split a dataset into a test, train and validations set. Using a datetime column. """
    
    assert len(X) == len(y) == len(dates)
    
    N = len(dates)
    dates = dates.sort_values(ascending=False)
    
    test_indx = dates.index[0:N//5 ]
    vali_indx = dates.index[N//5:2*N//5]
    train_indx = dates.index[2*N//5:]
    
    X_test = X.loc[test_indx , :]
    X_val = X.loc[vali_indx , :]
    X_train = X.loc[train_indx ,:]
    
    y_test = y.loc[ test_indx ]
    y_val = y.loc[ vali_indx ]
    y_train = y.loc[ train_indx ]
    
    return X_test, X_val, X_train, y_test, y_val, y_train




def get_weights( y: pd.Series , typ: str = 'balanced'):
    """ Create a set of weights to help class balance. """
    weights = compute_class_weight( typ , sorted(list(set(y))) , y )
    return  [ weights[x] for x in list(y) ]







