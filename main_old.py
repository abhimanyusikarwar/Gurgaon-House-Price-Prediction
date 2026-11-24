import pandas as pd
import numpy as np


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score



#1. Load The DataSet
housing = pd.read_csv("housing.csv", skipinitialspace=True)

#2. Split the DataSet into Train and Test Sets
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)# We Will Work With The COPY OF The Training DATA in future Steps
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)#Set Aside The Test Set 

#We Will Work With The COPY OF The Training DATA
housing = strat_train_set.copy()

#3. Separate Features and Labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)



#print(housing, housing_labels)

#4. Seperate Numerical and Categorical Columns
num_attribs = housing.drop("ocean_proximity", axis = 1).columns.tolist()
cat_attribs = ["ocean_proximity"]

#5. Lets Make The Pipeline For Numerical Attributes
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy ="median")),
    ('scalar', StandardScaler())
])

# Lets Make The Pipeline For Categorical Attributes
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown = "ignore"))
])

#Contruct The Full Pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs)
])

#6. Transform The Data
housing_prepared = full_pipeline.fit_transform(housing,housing_labels)
#print(housing_prepared.shape)

#7. Train The Model
#7.1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv =10)
#print(f"The Root Mean Squared Error for Linear Regression is: {lin_rmse}")
print(pd.Series(lin_rmses).describe())

#7.2. Decision Tree Regressor
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_pred)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv =10)
#print(f"The Root Mean Squared Error for Decision Tree Regressor is: {dec_rmse}")
print(pd.Series(dec_rmses).describe())

#7.3. Random Forest Regressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
forest_pred = forest_reg.predict(housing_prepared)
#forest_rmse = root_mean_squared_error(housing_labels, forest_pred)
#print(f"The Root Mean Squared Error for Random Forest Regressor is: {forest_rmse}")
forest_rmses = -cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error",cv =10)
#print(f"The Root Mean Squared Error for Random Forest Regressor is: {forest_rmse}")
print(pd.Series(forest_rmses).describe())
    