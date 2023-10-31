#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import neccessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# load the data
df = pd.read_csv('group_data_cleaning.csv')
df.head()


# In[3]:


df.info()


# ####  Feature Engineering

# In[4]:


df['What is your current monthly income level?'].unique()


# In[5]:


# Create the binary variable 'Income_Above_200K' using a conditional mapping
df['Income_Above_200K'] = df['What is your current monthly income level?'].apply(lambda x: 1 if 'and more' in x or '200,000' in x else 0)


# In[6]:


df['Income_Above_200K'].unique()


# In[7]:


df['Career Success'] = df['Income_Above_200K']


# In[8]:


# check for the unique levels of education
df['What is your highest level of education?'].unique()


# In[9]:


# Create a dictionary to map educational levels
education_levels = {
    'PhDs/Doctorate Degree': 6,
    'MBA degree': 5,
    "Master's degree": 4,
    "Bachelor's degree": 3,
    'Higher National Diploma': 2,
    'Ordinary National Diploma': 1
}

# Map the educational levels to a new feature
df['Educational Achievement Level'] = df['What is your highest level of education?'].map(education_levels)
df['Educational Achievement Level']


# In[10]:


df['Educational Achievement Level'].unique()


# In[11]:


df = df.dropna(subset=['Educational Achievement Level'])
df['Educational Achievement Level'].unique()


# In[12]:


#create an experience level
df['Graduation Year'] = pd.to_datetime(df['Year of graduation'], format='%Y')

# Calculate experience level by subtracting graduation year from the current year
current_year = pd.Timestamp('now')
df['Experience Level'] = current_year.year - df['Graduation Year'].dt.year
df['Experience Level']


# ##### Data Splitting

# In[13]:


# Define the target variable
target_column = 'Career Success'
y = df[target_column]

# Define the features or predictors
feature_columns = [
    'Year of graduation',
    'Educational Achievement Level',
    'Experience Level',
    'What sector/industry is your company in? (E.g. Banking, Agriculture, Telecommunication)',
    'What is your current monthly income level?',
]
X = df[feature_columns]

# Preprocess categorical features using label encoding
le = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = le.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a classification model (Random Forest in this example)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)


# In[14]:


df['Career Success'].unique()


# In[15]:


#split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the validation dataset
y_val_pred = model.predict(X_val)

# Calculate and print the evaluation metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
confusion = confusion_matrix(y_val, y_val_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(confusion)


# In[16]:


#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Lasso Regression with increased max_iter
lasso = Lasso(alpha=1.0, max_iter=10000)  # Increase max_iter
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Ridge Regression
ridge = Ridge(alpha=1.0)  # You can adjust the regularization strength (alpha)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("Linear Regression MSE:", mse_lr)
print("Lasso Regression MSE:", mse_lasso)
print("Ridge Regression MSE:", mse_ridge)


# In[17]:


# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared (R2):", r2)


# In[18]:


# Initialize and train the Random Forest Regressor model
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
random_forest_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = random_forest_reg.predict(X_test)

# Evaluate the Random Forest Regressor model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the evaluation metrics
print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R-squared (R2):", r2_rf)


# #### Evaluation Metrics

# In[19]:


y_true = y_test
# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R2)
r2 = r2_score(y_true, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_true, y_pred)

# Residual Plot
residuals = y_true - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Q-Q Plot
import statsmodels.api as sm
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot")
plt.show()


# #### Interpretations
# 

# Accuracy: The model correctly predicted all instances in the dataset.
# Precision:For class 0, the precision is 1.00, which means that all instances predicted as class 0 were correct.
# For class 1, the precision is also 1.00, indicating that all instances predicted as class 1 were correct.
# Recall:A recall of 1.00 for both classes indicates that the model correctly identified all instances of both classes.
# F1-Score:F1-score of 1.00 for both classes suggests a perfect balance between precision and recall for both classes
# Support:For class 0, there are 517 instances, and for class 1, there are 27 instances.
# 

# Linear Regression MSE: 0.047088619540719426
# For the Linear Regression model, the MSE is approximately 0.0471.
# This value indicates that, on average, the squared difference between the actual and predicted values is 0.0471.
# A lower MSE suggests that the model's predictions are closer to the actual values, which is a good sign of model performance.
# Lasso Regression MSE: 0.04759665096507353
# For the Lasso Regression model, the MSE is slightly higher, approximately 0.0476.
# This means that the Lasso model's predictions have a slightly higher average squared difference from the actual values compared to the Linear Regression model.
# The Lasso Regression model introduces L1 regularization, which can lead to feature selection and result in slightly different predictive performance compared to linear regression.
# Ridge Regression MSE: 0.04717483706282427
# For the Ridge Regression model, the MSE is approximately 0.0472.
# The Ridge model's performance falls between that of Linear Regression and Lasso Regression.

# Mean Squared Error (MSE):the MSE is approximately 0.0471. This means that, on average, the squared difference between the actual and predicted values is 0.0471.
# A lower MSE indicates that the model's predictions are closer to the actual values.
# 
