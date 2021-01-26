##################################################################################
## IMPORTS
##################################################################################

#print("{}\nIMPORTS ... \n{}".format("-"*35,"-"*35))

import os
import re
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

## Multivariate Feature Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

##################################################################################
## Data visualisation settings
##################################################################################
plt.style.use('seaborn')

# %matplotlib inline

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

#fig = plt.figure()
# Set up the matplotlib figure
#f, axs = plt.subplots(2,3,figsize=(20, 10))

##################################################################################
## FUNCTIONS DEFINITION
##################################################################################

#print("{}\nFUNCTIONS DEFINITION ... \n{}".format("-"*35,"-"*35))

##################################################################################
## Data loading
##################################################################################
'''
def load_data():
    ## load the data and print the shape
    df = pd.read_csv(os.path.join(DATA_DIR, "aavail-data-visualization.csv"), index_col=0)
    df.head(n=4)
    return df

## clean up the column names
#df.columns = [re.sub("\s+","_",col) for col in df.columns.tolist()]
'''
##################################################################################
## Data summary printing
##################################################################################

def print_data_summary(df):
    
    ## check the first few rows
    
    print("{}\nDATAFRAME SHAPE\n{}".format("-"*35,"-"*35))
    print("df: {} x {}".format(df.shape[0], df.shape[1]))

    print("{}\nDATAFRAME SUMMARY\n{}".format("-"*35,"-"*35))
    print(df.info())

    ## missing values summary
    print("\n{}\nMISSING VALUE SUMMARY\n{}".format("-"*35,"-"*35))
    print(df.isnull().sum(axis = 0))
    
    print("\n{}\n'ZERO' VALUE SUMMARY\n{}".format("-"*35,"-"*35))
    print((df == 0).sum(axis = 0))
'''
    df1 = df[df['subscriber_type'].isnull()]
    print("\n{}\nNumber of 'subscriber_type' missing per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df1, index= ["country_name"], values="customer_name",aggfunc='count').round(3))

    df2 = df[df['num_streams'].isnull()]
    print("\n{}\nNumber of 'num_streams' missing per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df2, index= ["country_name"], values="customer_name",aggfunc='count').round(3))

    df3 = df[(df['num_streams'].isnull())& (df['subscriber_type'].isnull())]
    print("\n{}\nNumber of records missing 'num_streams & subscriber_type' per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df3, index= ["country_name"], values="customer_name",aggfunc='count').round(3))

    print("\n{}\nEXPLORATORY DATA ANALYSIS\n{}".format("-"*50,"-"*50))

    columns_to_show = ["age","num_streams"]
    
    print("\n{}\nNumber of 'dates' lines per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df, index= ["country"], values="dates",aggfunc='count').round(3))
    
    print("\n{}\nNumber of records per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df, index= "country_name", values="customer_name",aggfunc='count').round(3))

    print("\n{}\nAverage age and num_stream per country\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df, index= "country_name", values=columns_to_show,aggfunc='mean').round(3))

    print("\n{}\nAverage age and num_stream per subscriber_type\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df, index= "subscriber_type", values=columns_to_show,aggfunc='mean').round(3))

    print("\n{}\nNumber of churns per country and per subscriber_type\n{}".format("-"*50,"-"*50))
    print(pd.pivot_table(df, index= ["country_name","subscriber_type","is_subscriber"], values="customer_name",aggfunc='count').round(3))

##################################################################################
## Program start ...
##################################################################################
print("\n{}\nProgram start ...\n{}".format("-"*50,"-"*50))
## specify the directory you saved the data and images in
DATA_DIR = os.path.join("..","data")
IMAGE_DIR = os.path.join("..","images")
df = load_data()
print_data_summary(df)

##################################################################################
## drop the rows that have NaNs
##################################################################################
print("\n{}\nDrop the rows that have NaNs\n{}".format("-"*50,"-"*50))
print("Original Matrix:", df.shape)
original_nb_lines = df.shape[0]

df.dropna(inplace=True)
print("After NaNs removed:", df.shape)
dropna_na_lines = df.shape[0]

print("Number of lines removed: ", original_nb_lines - dropna_na_lines)

##################################################################################
## drop the columns that have NaNs
##################################################################################
print("\n{}\nDrop the columns that have NaNs\n{}".format("-"*50,"-"*50))
df = load_data()
## drop the columns that have NaNs
print("Original Matrix:", df.shape)
original_nb_cols = df.shape[1]
df.dropna(inplace=True, axis='columns')
dropna_na_cols = df.shape[1]
print("After NaNs removed:", df.shape)
print("Number of columns removed: ", original_nb_cols - dropna_na_cols)


##################################################################################
## In order to do imputation, data is transformed
## 'subscriber_type' data transformation: aavail_premium => 1, aavail_basic => 2, aavail_unlimited => 3
## 'country_name' data transformation: One hot encoding 
## 'customer_name' removed
##################################################################################

df = load_data()
columns = ['subscriber_type','is_subscriber','age','num_streams','country_name']
print("{}\nINITIAL DATAFRAME\n{}".format("-"*35,"-"*35))
print(df[columns].head(n=6))

print("\n{}\n1) 'subscriber_type' data transformation: aavail_premium => 1, aavail_basic => 2, aavail_unlimited => 3".format("-"*100,"-"*100))
df['subscriber_type'] = df['subscriber_type'].replace('aavail_premium',1.0)
df['subscriber_type'] = df['subscriber_type'].replace('aavail_basic',2.0)
df['subscriber_type'] = df['subscriber_type'].replace('aavail_unlimited',3.0)


########################################################

print("2)'country_name' data transformation: One hot encoding\n3) 'customer_name' removed\n{}".format("-"*100,"-"*100))

#One hot encode the country_name
ohe1 = OneHotEncoder()
column = df['country_name'].values.reshape(-1,1)
ohe1.fit(column)
labels1 = ohe1.categories_[0].tolist()
X1 = ohe1.transform(column).toarray()


##Concat all of the data
labels = ['subscriber_type','is_subscriber','age','num_streams']
X = df.loc[:,labels].to_numpy()
labels = labels + labels1
X = np.hstack([X,X1])
df1 = pd.DataFrame({label:X[:,i] for i,label in enumerate(labels)})
print(df1.head(n=6))

print("{}\nORIGINAL DATAFRAME DATA TYPES\n{}".format("-"*35,"-"*35))
print(df.dtypes)
print("{}\nTRANSFORMED DATAFRAME DATA TYPES\n{}".format("-"*35,"-"*35))
df1 = df1.infer_objects()
print(df1.dtypes)
print("{}\nBEFORE IMPUTATION: 'Subscriber_type' and 'num_streams' description\n{}".format("-"*60,"-"*60))
print(df1[['subscriber_type','num_streams']].describe())

##################################################################################
## Imputation of missing values: Multivariate feature imputation
##################################################################################

print("{}\nIMPUTATION OF MISSING VALUES\n{}".format("-"*35,"-"*35))
strategies = ["most_frequent", "constant","mean", "median"]
for s in strategies:
    print("{}\nMultivariate feature imputation: Strategy = {}\n{}".format("-"*60,s,"-"*60))
    imp = IterativeImputer(max_iter=10, random_state=0,initial_strategy=s)
    imp.get_params(deep=True)
    df_with_imputed_values = imp.fit_transform(df1)
    c=['subscriber_type','is_subscriber','age','num_streams','singapore','united_states']
    df_with_imputed_values = pd.DataFrame(data=df_with_imputed_values, index=None, columns=c)
    print(df_with_imputed_values[['subscriber_type','num_streams']].describe())
    
##################################################################################
## Data visualisation
##################################################################################
print("{}\nDATA VISUALISATION\n{}".format("-"*35,"-"*35))
##################################################################################

print("{}\n1) 'subscriber_type' reverse data transformation: 1 => aavail_premium , 2 => aavail_basic, 3 => aavail_unlimited\n{}".format("-"*100,"-"*100))
df1['subscriber_type'] = df1['subscriber_type'].replace(1.0,'aavail_premium')
df1['subscriber_type'] = df1['subscriber_type'].replace(2.0,'aavail_basic')
df1['subscriber_type'] = df1['subscriber_type'].replace(3.0,'aavail_unlimited')
print(df1.head(n=6))

print("{}\n1) 'country_name' reverse data transformation: Reverse one-encoding\n{}".format("-"*100,"-"*100))

def get_country(row):
    return(row.index[row.apply(lambda x: x==1)][0])

# prepare a country column
df_tmp = pd.concat([df1['singapore'], df1['united_states']], axis=1)
df_tmp = df_tmp.apply(lambda row:get_country(row), axis=1)
df1 = pd.concat([df1, df_tmp], axis=1)
df1 = df1.drop(columns=['singapore', 'united_states'])
df1 = df1.rename({0: 'country_name'}, axis='columns',errors="raise")
print(df1.head(n=6))

print("{}\nIMPUTED DATAFRAME DATA TYPE\n{}".format("-"*35,"-"*35))
print(df1.dtypes)

print("{}\nIMPUTED DATAFRAME DATA TYPE UPDATED WITH CATEGORIES\n{}".format("-"*35,"-"*35))
df1['subscriber_type'] = df1['subscriber_type'].astype('category')
df1['country_name'] = df1['country_name'].astype('category')
print(df1.dtypes)


sns.countplot(x='is_subscriber',data=df1, hue='country_name',ax=axs[0]).set_title("Number of churns per country")
sns.catplot(x="country_name", y="age", hue='subscriber_type', kind="box", data=df1,ax=axs[1]).set(title = "Age of subscriber per subscriber type and per country")
sns.catplot(x="country_name", y="num_streams", hue='subscriber_type', kind="box",data=df1,ax=axs[2]).set(title = "Number of streams per subscriber type and per country")
sns.catplot(x="country_name", y="age", hue='is_subscriber', kind="box", data=df1,ax=axs[3]).set(title = "Age of subscriber per country")
sns.catplot(x="country_name", y="num_streams", hue='is_subscriber', kind="violin",data=df1,ax=axs[4]).set(title = "Number of streams per subscriber and per country")

sns.barplot(x='country_name',y='is_subscriber',data=df1, ax=axs[0,0]).set_title("Number of churns per country")
sns.boxplot(x="country_name", y="age", data=df1,ax=axs[0,1]).set(title = "Age of subscriber per country")
sns.boxplot(x="country_name", y="num_streams", data=df1,ax=axs[0,2]).set(title = "Number of streams per country")
sns.scatterplot(x="age", y="num_streams", data=df1,ax=axs[1,0]).set(title = "Number of streams per Age")
sns.pointplot(x="country_name", y="is_subscriber", data=df1,ax=axs[1,1]).set(title = "Number of streams per country")
sns.barplot(x='subscriber_type',y='age',data=df1, ax=axs[1,2]).set_title("Age per Subscriber type")

## make a pair plot
columns = ['is_subscriber','age','num_streams']
#axes = sns.pairplot(df1,vars=columns,hue="country_name",palette="husl")

image_path = os.path.join(IMAGE_DIR,"07112020_visualisation_assignment.png")    
plt.savefig(image_path,bbox_inches='tight',pad_inches = 0,dpi=500)
print("{} created.".format(image_path))
'''