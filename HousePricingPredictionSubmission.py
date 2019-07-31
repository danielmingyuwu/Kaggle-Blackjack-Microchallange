import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *

def find_the_bad_columns(SumColumns:list):
	"""
	Given the a list, which contained the number of null value
	in each columns. Return the positions of bad columns.
	"""
	new_list = []
	i = 0
	for a in SumColumns:
		if a == 0:
			i += 1
		else:
			new_list.append(i)
			i += 1
	return new_list



OriginalDataSet = '/Users/danielwu/Documents/GitHub/Kaggle-Learn-HousingPriceModel/train_data.csv'

OriginalData = '/Users/danielwu/Documents/GitHub/Kaggle-Learn-HousingPriceModel/train_data.csv'
row_home_data = pd.read_csv(OriginalData)
#Load the data
row_home_data.describe()
row_home_data.head()
row_home_data.tail()
#check the main features of data
row_home_data.shape()
#check the size of the matrix
All_features = row_home_data.columns
#However, not all the columns contain all true value. Some
#may have null value, which we don't want.

#Find all columns that contains null value
NumOfNull = row_home_data.isnull().sum().tolist()
ListWithBadColumns = find_the_bad_columns(NumOfNull)
Selected_features = row_home_data.columns[ListWithBadColumns]
print(Selected_features) #check
#Since we know the columns with null value, we can check how
#much the null value influences our data.
Row_PercentageError = (row_home_data.isnull().sum() / len(row_home_data)) * 100
PercentageError = Row_PercentageError[PercentageError != 0]
print(Selected_features)
#Because null data will cause imprecise prediction, we may
#modify or change the data when we build the model.
#Because our predicton only focues on the SalePrice of the
#house, we set the target object as y
y = row_home_data.SalePrice

print('yes')





