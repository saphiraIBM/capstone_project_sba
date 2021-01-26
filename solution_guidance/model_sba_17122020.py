import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

import statsmodels.api as sm
from sklearn.svm import SVC

from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR

from sklearn.metrics import classification_report

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
#MODEL_VERSION_SVC = 0.2
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def _model_train(df,tag,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """


    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       shuffle=True, random_state=42)
                                                       
    ###################################################################################
    ## train a random forest model  
    ###################################################################################
    print("\nTRAINING MODELS: RANDOM FOREST MODEL")
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])
    
    grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1,return_train_score=True)
    grid_rf.fit(X_train, y_train)
    scores_df_rf = pd.DataFrame(grid_rf.cv_results_).sort_values(by='rank_test_score')
    scores_df_rf['model']=grid_rf
    scores_df_rf = scores_df_rf[scores_df_rf['rank_test_score'] == 1]
    
    #print(scores_df_rf.columns)
    print(grid_rf.best_params_)
    print(grid_rf.best_score_)
    
    y_pred = grid_rf.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    scores_df_rf['eval_rmse']=eval_rmse
    print ("eval_rmse: {}".format(eval_rmse))
    #print(scores_df_rf[['rank_test_score','params','mean_test_score','mean_fit_time','mean_score_time','eval_rmse']])
    print("\nEND OF TRAINING MODELS: RANDOM FOREST MODEL")
    
    ###################################################################################
    ## train a bagging model  
    ###################################################################################
    print("\nTRAINING MODELS: BAGGING MODEL")
    ## train a bagging model
    pipe_bag = Pipeline(steps=[('scaler', StandardScaler()),
                               ('bag', BaggingRegressor(base_estimator=SVR(), random_state=0))])
    
    param_grid_bag = {
    'bag__n_estimators': [10,15,20,25]
    }
    
    grid_bag = GridSearchCV(pipe_bag, param_grid=param_grid_bag, cv=5, iid=False, n_jobs=-1)
    grid_bag.fit(X_train, y_train)
    #print(grid_bag.get_params())
    #print(grid_bag.score(X_train, y_train))
    #print(grid_bag.cv_results_)
    scores_df_bag = pd.DataFrame(grid_bag.cv_results_).sort_values(by='rank_test_score')
    scores_df_bag['model']=grid_bag
    scores_df_bag = scores_df_bag[scores_df_bag['rank_test_score'] == 1]
    #print(scores_df_bag.columns)
    print(grid_bag.best_params_)
    print(grid_bag.best_score_)
        
    y_pred = grid_bag.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    scores_df_bag['eval_rmse']=eval_rmse
    print ("eval_rmse: {}".format(eval_rmse))
    #print(scores_df_bag[['rank_test_score','params','mean_test_score','mean_fit_time','mean_score_time','eval_rmse']])
    print("\nEND OF TRAINING MODELS: BAGGING MODEL")
    
    ## Compare models
    results_df = scores_df_rf.append(scores_df_bag,ignore_index=True).sort_values(by='mean_test_score',ascending=False)
    print(results_df[['model','params','mean_test_score','mean_fit_time','mean_score_time','eval_rmse']])
    
    best_model = results_df['model'].loc[0]
    print("best_model: {}".format(best_model))
    
    ## retrain using all data and the Random Forest model
    best_model.fit(X, y)
     
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(best_model,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
    '''update_train_log((str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)'''
  

def model_train(data_dir,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        #if test and country not in ['all','united_kingdom']:
        if test and country not in ['all']:
            continue
        
        _model_train(df,country,test=test)
    
def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """
    print("...... model load")
    
    if not data_dir:
        data_dir = os.path.join("data","cs-train")
        #data_dir = os.path.join("..","data","cs-train")
    
    #prefix = ""
    #models = [f for f in os.listdir(os.path.join(".","models")) if re.search("sl",f)]
    models = [f for f in os.listdir(os.path.join(".","models")) if re.search(prefix,f)]
    #models = [f for f in os.listdir(os.path.join("models")) if re.search(prefix,f)]
    print("models: {}".format(models))
    

    if len(models) == 0:
        print ("Models with prefix {} cannot be found did you train?".format(prefix))
        raise Exception("Models with prefix {} cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))
    print("all_models keys: {}".format(all_models.keys()))
    
    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data,all_models = model_load(training=True)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    print("country: {}".format(country))
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)
    
    #decompose the data into four different components: Observed,Trended,Seasonal,Residual
    #seasonal_decompose(y_pred)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)
    '''update_predict_log(y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)'''


    return({'y_pred':y_pred,'y_proba':y_proba})



# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()
    
if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("\nTRAINING MODELS")
    data_dir = os.path.join("data","cs-train")
    model_train(data_dir,test=True)

    
    ## load the model
    print("\nLOADING MODELS")
    all_data, all_models = model_load()
    print("... models loaded: ",",".join(all_models.keys()))
    
    ## test predict
    print("\nPREDICTING MODEL FOR ALL")
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day)
    print(result)
    
    ## test predict 2
    print("\nPREDICTING MODEL FOR UK")
    country='united_kingdom'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day)
    print(result)
    
    
