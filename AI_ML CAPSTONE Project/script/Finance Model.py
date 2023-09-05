#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel


# In[2]:


df = pd.read_excel('../data/Reporte_de_Incentivos_GM_AI.xlsx', header=0)
df = df.fillna(0)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# Remove Non-needed or irrelevant Columns
columns_to_drop = ['Body Type','Transmission', 'Doors', 'Promotion 1','Promotion 2','Promotion 3','Name Plan']
df_dropped = df.drop(columns=columns_to_drop)

# Now I'll drop datetime cells for testing purposes:
datetime_cols = ['Valid Since', 'Valid Until', 'Date']
df_dropped = df_dropped.drop(columns=datetime_cols)

df_dropped.info()


# In[6]:


#Now filter our DF based on the Segment. This will allow us to get more accurate predictions, based on each segment.
segment_fiter = df_dropped['GM Segment'] == 'Car-B'
df_segment = df_dropped[segment_fiter]
#df_segment.info()


# In[7]:


# Now let's drop our Categorical columns that indicate the model of our segment:
model_cols = ['MODEL/SEGMENT', 'Make', 'Model', 'Version', 'UID','GM Segment']
df_segment = df_segment.drop(columns=model_cols)
df_segment.info()


# In[8]:


# "Fix" the Boolean column Y/N to 1/0
df_segment['Subsidized Plan'] = df_segment['Subsidized Plan'].replace({'Y':1 , 'N':0})

# Perform one-hot encoding on categorical variables
categorical_cols = ['Plan Type']
X_encoded = pd.get_dummies(df_segment, columns=categorical_cols)


# In[11]:


# Separate features and target variable
X = X_encoded.drop('Volumen', axis=1)
y = X_encoded['Volumen']


# In[9]:


# Let's check our CORR Matrix.
# Create the correlation matrix
corr = df_segment.corr()

# Create the heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[12]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


model = LinearRegression().fit(X_train_scaled, y_train)
model.score(X_test_scaled, y_test)


# In[292]:


# Now I'll apply Permutation Feature.
# Permutation feature importance is a strategy for inspecting a model and its features importance.

model = LinearRegression().fit(X_train_scaled, y_train)
model.score(X_test_scaled, y_test)

from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test_scaled, y_test,
                            n_repeats=30,
                            random_state=0)

importances = []
for i in r.importances_mean.argsort()[::-1]:
    col_name = X_test.columns[i]
    var = (r.importances_mean[i])
    importances.append({
        "Feature":col_name,
        "Importances_Mean":float(var)
    })
    
importances


# In[293]:


# Extract the feature names and importances from the importances array
feature_names = [x["Feature"] for x in importances]
importances_mean = [x["Importances_Mean"] for x in importances]

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(feature_names, importances_mean)

# Customize the plot
ax.set_title("Feature Importances")
ax.set_xlabel("Features")
ax.set_ylabel("Importances")
ax.tick_params(axis="x", rotation=90)
plt.show()


# In[208]:


# SequentialFeatureSelector with Lasso Regression
sfs = SequentialFeatureSelector(Lasso(alpha=0.1), n_features_to_select=4)
sfs.fit(X_train, y_train)
print("SequentialFeatureSelector with Lasso selected features:\n", X_train.columns[sfs.get_support()])


# In[287]:


# RFE with Lasso Regression
rfe = RFE(Lasso(alpha=0.1), n_features_to_select=4)
rfe.fit(X_train, y_train)
print("RFE with Lasso selected features:\n", X_train.columns[rfe.support_])


# In[211]:


model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)


# In[97]:





# In[102]:





# In[ ]:


import seaborn as sns

# Drop ocean_proximity and total_bedrooms columns
cali = cali.drop(['ocean_proximity', 'total_bedrooms'], axis=1)

# Create the correlation matrix
corr = cali.corr()

# Create the heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

