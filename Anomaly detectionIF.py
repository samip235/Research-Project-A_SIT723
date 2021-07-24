#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd


# In[23]:


df=pd.read_csv(r"C:\Users\samip\Downloads\kddcup99.csv")

df.head()


# In[24]:


data_df=pd.pivot_table(df,values='duration',index='dst_host_srv_count',columns='protocol_type')
data_df.reset_index(inplace=True)
data_df.fillna(0,inplace=True)
data_df


# In[25]:


data_df.columns
#specify the 12 metrics column names to be modelled
to_model_columns=data_df.columns[1:13]
from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12),                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(data_df[to_model_columns])
pred = clf.predict(data_df[to_model_columns])
data_df['anomaly']=pred
outliers=data_df.loc[data_df['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(data_df['anomaly'].value_counts())


# In[26]:


#Normalize and fit the metrics to a PCA to reduce the number of dimensions and then plot them in 3D highlighting the anomalies
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)  # Reduce to k=3 dimensions
scaler = StandardScaler()
#normalize the metrics
X = scaler.fit_transform(data_df[to_model_columns])
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
           lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()


# In[27]:


#Plotting the same fed to a PCA reduced to 2 dimensions
from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(data_df[to_model_columns])
res=pd.DataFrame(pca.transform(data_df[to_model_columns]))
Z = np.array(res)
plt.title("IsolationForest")
plt.contourf( Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(res[0], res[1], c='green',
                 s=20,label="normal points")
b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
plt.legend(loc="upper right")
plt.show()


# In[ ]:





# In[ ]:




