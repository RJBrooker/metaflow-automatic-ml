# Import packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from metaflow import FlowSpec, step, IncludeFile
import glob ,sys , os
import featuretools as ft
import pandas as pd
from tqdm import tqdm
import numpy as np
import swifter 
from utils import infer_column_types, random_dates, test_train_temporal_split, get_weights
from tqdm import tqdm
import feather
import shap
import cloudpickle


from functools import wraps
from typing import Callable, Any,Optional






class ModelPipeline(FlowSpec):
    """
    The flow performs the following steps:
    """
    
    
    @step
    def start(self):
        """ 01. Initialise the data paths and transformation functions.  """
        self.data_dir = '../data/raw_data'
        self.trans_primitives = [ 'weekday'  , 'hour', 'time_since_previous'] 
        self.agg_primitives = ['mean', 'max', 'min', 'std', 'count',  'percent_true', 'last', 'time_since_last', 'mode']
        self.ignore_cols =  ['num_contacts', 'num_referrals', 'num_successful_referrals' ]
        self.feature_windows = [10, 30, 60 , 90 ] #[10,20,30]
        self.max_feature_depth = 2
        self.next( self.load_raw_data )
    
    
    
    
    @step
    def load_raw_data(self):
        """ 02. Load the dataset.  """
        
        # Load the datasets
        self.users = pd.read_csv(f'{self.data_dir}/users.csv', parse_dates=['created_date'] ).set_index('user_id').drop(columns=self.ignore_cols)
        self.transactions = pd.read_csv(f'{self.data_dir}/transactions.csv', parse_dates=['created_date']).set_index('user_id')
        self.notifications = pd.read_csv(f'{self.data_dir}/notifications.csv', parse_dates=['created_date']).set_index('user_id')
        self.next( self.generate_cutoff_datetimes )
    
    
    
    @step
    def generate_cutoff_datetimes(self):
        """ 03. Give each users a random "cut-off datetime". This is the datetime we are observing our users at. """
        
        n_users = len(self.users)
        max_date = self.transactions.created_date.max()
        self.users['cut_off_ds'] = random_dates( max_date - pd.to_timedelta( 30+90 , unit='d'), max_date - pd.to_timedelta( 30 , unit='d'),n_users  )
        self.users = self.users.sort_values('cut_off_ds', ascending=False)
        # append to transactions
        for name in ['transactions']:
            getattr(self,name)['cut_off_ds'] = self.users['cut_off_ds']
            getattr(self,name)['days_before_cutoff'] = ( getattr(self,name)['created_date'] - getattr(self,name)['cut_off_ds'] ).dt.days
        self.next( self.generate_target_variable )
    
    
    
    
    
    @step
    def generate_target_variable(self):
        """ 04. Calculate the target variable for the 30 day period after the user "cut-off datetime"."""
        
        # This is our target variable 
        self.label = (self.users.assign(
            label = self.transactions.query(' 0 > days_before_cutoff >= -30 ').groupby('user_id').amount_usd.sum() 
        ).label.fillna(0) > 0).astype(int)
        self.next( self.generate_entity_set )
    
    
    
    
    @step
    def generate_entity_set(self):
        """ 05. Define the entity set along with the table relations. """
        
        import featuretools as ft
        self.es = ft.EntitySet(id = 'clients')
        self.es = self.es.entity_from_dataframe( entity_id = 'users', dataframe = self.users.reset_index() ,  index = 'user_id', time_index = 'created_date')
        
        for d in self.feature_windows :
            self.es = self.es.entity_from_dataframe(
                entity_id = f'transactions_{d}d', 
                dataframe = self.transactions.query(f' {d}  > days_before_cutoff >= 0  ').reset_index(),
                index = 'transaction_id', time_index = 'created_date')
            # Add the relationship between customera and transactions
            self.es = self.es.add_relationship(ft.Relationship(self.es['users']['user_id'], self.es[f'transactions_{d}d']['user_id']))
        
        self.next( self.generate_features )
    
    
    
    
    
    @step 
    def generate_features(self):
        """ 06. Run deep feature synthesis .  """
        
        # Create new features using specified primitives
        self.feature_matrix, self.feature_defs = ft.dfs( 
            entityset = self.es, target_entity = 'users',  trans_primitives = self.trans_primitives , agg_primitives = self.agg_primitives , verbose=1 , max_depth=self.max_feature_depth  )
        
        # encode at 
        self.feature_matrix_enc, self.features_enc = ft.encode_features(self.feature_matrix, self.feature_defs)        
        self.next( self.split_training_data )
    
    
    
    @step 
    def split_training_data(self):
        """ 07. Create the training, test and validation set. """
        self.X_test, self.X_val, self.X_train, self.y_test, self.y_val, self.y_train = (
            test_train_temporal_split(self.feature_matrix_enc.fillna(-10**9), self.label , self.users['cut_off_ds']) )
        self.next( self.generate_sample_weights )
    
    
    
    @step 
    def generate_sample_weights(self):
        """ 08. Create sample weights based on class balance.  """
        self.w_train = get_weights(  self.y_train  )
        self.w_val = get_weights(  self.y_val  )
        self.next( self.fit_model )
    
    
    
    @step 
    def fit_model(self):
        """ 09. Fit natural gradient boosting model. """
        
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        
        mdl = NGBClassifier(Dist=Bernoulli)
        mdl.fit( self.X_train, self.y_train , X_val=self.X_val, Y_val=self.y_val ,  sample_weight=self.w_train, val_sample_weight=self.w_val, early_stopping_rounds=100)
        self.mdl = cloudpickle.dumps(mdl)
        self.next( self.compute_roc )
    
    
    
    
    def mdl_(self):
        """ A hack to fixes the pickling problem; """
        return cloudpickle.loads(self.mdl)
    
    
    
    
    
    @step 
    def compute_roc(self):  
        """ 10. Calculate AUC scores. """
        from sklearn.metrics import roc_auc_score
        
        self.test_preds = self.mdl_().pred_dist(self.X_test).probs[1]
        self.train_preds = self.mdl_().pred_dist(self.X_train).probs[1]
        
        self.test_roc = roc_auc_score(self.y_test.astype(int) , self.test_preds)
        self.train_roc = roc_auc_score(self.y_train.astype(int) , self.train_preds)
        
        self.next( self.generate_shap_values )
    
    
    
    @step 
    def generate_shap_values(self):
        """ 11. Generate shap values. """
        explainer = shap.TreeExplainer(self.mdl_(), model_output=0) # use model_output = 1 for scale trees
        self.shap_values = explainer.shap_values(self.X_train)
        self.next( self.end )
    
    
    
    @step
    def end(self):
        """
        End the flow.
        """
        pass
    






if __name__ == '__main__':
    ModelPipeline()





