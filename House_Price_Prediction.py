#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# In[2]:


# Load CSV
df = pd.read_csv("/workspaces/House_Price_Prediction/data (1).csv")

print("Shape:", df.shape)
df.head()


# In[15]:


df.info()


# In[17]:


# Remove rows with missing values
df.dropna(inplace=True)


# In[3]:


# Drop duplicates
df = df.drop_duplicates()

# Convert date column if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.drop(columns=["date"])

# Standardize categorical fields
for col in ["street","city","statezip","country"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Check missing values
print(df.isnull().sum().sort_values(ascending=False).head(10))


# In[14]:


# Correlation matrix for numeric features only
corr_matrix = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Quick look at correlation with target 'price'
if "price" in corr_matrix.columns:
    print("Correlation with price:")
    print(corr_matrix["price"].sort_values(ascending=False))


# In[18]:


# Summary statistics for numerical features
# This provides mean, std, min, max, etc., to understand data distribution
df.describe()


# In[4]:


TARGET = "price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train/Validation/Test split (60/20/20)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# In[22]:


# Distribution of the target variable (price)
# Visualize to check for skewness; house prices often have a long tail
plt.figure(figsize=(10, 6))
sns.histplot(y_train, bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[5]:


categorical_features = [c for c in X.columns if df[c].dtype == "object"]
numeric_features = [c for c in X.columns if c not in categorical_features]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_features),
    ("cat", cat_pipe, categorical_features)
])


# In[6]:


rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

# Validation predictions
y_val_pred = rf_model.predict(X_val)


# In[8]:


# Validation predictions 
y_val_pred = rf_model.predict(X_val)  
# Test predictions 
y_test_pred = rf_model.predict(X_test) 

# Normalized RMSE by target range 
val_range = np.max(y_val) - np.min(y_val) 
test_range = np.max(y_test) - np.min(y_test) 

val_rmse = (mean_squared_error(y_val, y_val_pred) ** 0.5) / val_range 
test_rmse = (mean_squared_error(y_test, y_test_pred) ** 0.5) / test_range

print(f"Validation Normalized RMSE: {val_rmse:.4f}") 
print(f"Test Normalized RMSE: {test_rmse:.4f}")


# In[21]:


# Example input (adjust values to your dataset)
sample = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1500,
    "sqft_lot": 4000,
    "floors": 1,
    "waterfront": 0,
    "view": 2,
    "condition": 3,
    "sqft_above": 1300,
    "sqft_basement": 200,
    "yr_built": 2005,
    "yr_renovated": 0,
    "street": "main st",
    "city": "seattle",
    "statezip": "wa 98103",
    "country": "usa",
    "year": 2014,
    "month": 6
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample])

# Predict price
pred_price = rf_model.predict(sample_df)[0]
print("Predicted price:", round(pred_price, 2))

# Save the trained model with pickle
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




