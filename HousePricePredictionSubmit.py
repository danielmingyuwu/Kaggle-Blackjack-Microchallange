import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
#from learntools.core import *

def quality_check (X_train, X_valid, y_train, y_valid):
    """
    Check the quality of predictions by MAE.
    """
    MODEL = RandomForestRegressor(n_estimators=10,random_state=1)
    MODEL.fit(X_train,y_train)
    predictions1 = MODEL.predict(X_valid)
    return mean_absolute_error(y_valid,predictions1)

#Load the data
OriginalData = '/Users/danielwu/Documents/GitHub/Kaggle-Learn-HousingPriceModel/train_data.csv'
row_home_data = pd.read_csv(OriginalData)

#Briefly check the data
print(row_home_data.head())
row_home_data.tail()
row_home_data.describe()


#Normal we set the target value as y
y = row_home_data.SalePrice
#We assign the possible factors of affecting the house price as x
X = (row_home_data.drop(['SalePrice'],axis=1)).select_dtypes(exclude=['object'])
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


#Some columns may contain null value, which causes imprecise prediction for future

cols_with_missing_values = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
print(cols_with_missing_values)
#There are many ways to deal with missing values. We can try two ways to deal with:
#either dropping the columns with missing value or we use imputation.
#Method 1:drop the columns
reduced_X_train = X_train.drop(cols_with_missing_values,axis=1)
reduced_X_val = X_val.drop(cols_with_missing_values,axis = 1)
print((quality_check(reduced_X_train, reduced_X_val, y_train, y_val)))
#Return MAE 18147.72

#Method 2:imputation
imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train))
X_val_imputed = pd.DataFrame(imputer.transform(X_val))
X_train_imputed.columns = X_train.columns
X_val_imputed.columns = X_val.columns

print(quality_check(X_train_imputed, X_val_imputed, y_train, y_val))
#Return MAE 18544.034520547943

#It turns out that MAE1 < MAE2, so method 1 is better. So we can drop the columns
#of missing values, which are 'LotFrontage', 'MasVnrArea', 'GarageYrBlt'.

#Except of missing data, we also need to deal with categorical data.
#Get rid of categorical data:
print(row_home_data.dtypes == 'object')

#Cteat a new variable features
features = ['LotArea','MSSubClass','OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd','1stFlrSF', '2ndFlrSF','KitchenAbvGr', 'TotRmsAbvGrd','YrSold','MoSold','MiscVal','PoolArea','ScreenPorch','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch']

X1 = row_home_data[features]
train_X1, val_X1, train_y,val_y = train_test_split(X, y, random_state=1)

FirstModel = DecisionTreeRegressor(random_state=1)
FirstModel.fit(train_X1,train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = FirstModel.predict(val_X1)
val_mae = mean_absolute_error(val_predictions, val_y)


# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X1, train_y)
val_predictions = iowa_model.predict(val_X1)
MAE_val = mean_absolute_error(val_predictions, val_y)


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X1, train_y)
rf_val_predictions = rf_model.predict(val_X1)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



#CREATE A MODEL FOR THE COMPETITION
RandomForestModelOnData = RandomForestRegressor(random_state = 1)
RandomForestModelOnData.fit(X1,y)

#MAKE PREDICTIONS
Test_X = row_home_data[features]
Submit_Prediction = RandomForestModelOnData.predict(Test_X)


#Below are required for Kaggle website for submission. 
#output = pd.DataFrame({'Id': test_data.Id,
                       #'SalePrice': test_preds})
#output.to_csv('submission.csv', index=False)































