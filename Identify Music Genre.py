#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

df = pd.read_csv("C:\\Users\\Chong Bin Yong\\Downloads\\AUG2022\\PRG4206\\Coursework\\Individual Assignment\\music_genre.csv")


# In[2]:


df.info()


# In[3]:


df.head()


# In[4]:


df.keys()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


print("music_genre:",len(df['music_genre'].unique()),"\b variables")


# In[9]:


y = df['music_genre']
print(y)


# In[10]:


df = df.drop(columns = ['instance_id','artist_name', 'track_name', 'popularity', 'duration_ms', 'obtained_date', 'valence', 'music_genre'])
df.head()


# In[11]:


numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[12]:


categorical = [col for col in df.columns if df[col].dtypes == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[13]:


print("key:",len(df['key'].unique()),"\b variables")
print("mode:",len(df['mode'].unique()),"\b variables")
print("tempo:",len(df['tempo'].unique()),"\b variables")


# In[14]:


le = preprocessing.LabelEncoder()
key_encoded = le.fit_transform(df['key'])
print (key_encoded)


# In[15]:


df['key'] = key_encoded
df['key']


# In[16]:


mode_encoded = le.fit_transform(df['mode'])
print (mode_encoded)


# In[17]:


df['mode'] = mode_encoded
df['mode']


# In[18]:


tempo_list = df['tempo'].tolist()
print(tempo_list)


# In[19]:


df = df[df.tempo != '?']
df.shape


# In[20]:


df['tempo'] = df['tempo'].str.replace('','').astype(np.float64)
df['tempo'].dtypes


# In[21]:


music_genre_encoded = le.fit_transform(y)
print (music_genre_encoded)


# In[22]:


y = music_genre_encoded
y


# In[23]:


numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[24]:


categorical = [col for col in df.columns if df[col].dtypes == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[25]:


df.isnull().sum()


# In[26]:


df.head()


# In[27]:


scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(df))
scaled_data


# In[28]:


km = KMeans(n_clusters = 3)
km


# In[29]:


y_predicted = km.fit_predict(scaled_data)
y_predicted


# In[30]:


df['cluster'] = y_predicted
df.head()


# In[31]:


k_rng = range(1,15)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(scaled_data)
    sse.append(km.inertia_)
    
sse


# In[32]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)


# In[33]:


k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km_labels = km.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, km_labels)
    print("For k =", k, ", the silhouette score is :", score,)


# In[34]:


model = KMeans(random_state=0) 
visualizer = KElbowVisualizer(model, k=(2,15), metric='silhouette', timings=False)

visualizer.fit(scaled_data)    
visualizer.poof()


# In[35]:


km = KMeans(n_clusters = 10)
km


# In[36]:


y_predicted = km.fit_predict(scaled_data)
y_predicted


# In[37]:


df['cluster'] = y_predicted
df.head()


# In[38]:


sns.heatmap(scaled_data.corr())


# In[39]:


pca = PCA(n_components = 2)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)

data_pca = pd.DataFrame(data_pca, columns = ['PC1', 'PC2'])
data_pca.head()


# In[40]:


sns.heatmap(data_pca.corr())


# In[41]:


km = KMeans(n_clusters = 3)
km


# In[42]:


y_predicted = km.fit_predict(data_pca)
y_predicted


# In[43]:


pca_df = pd.concat([data_pca, pd.DataFrame(y_predicted, columns = ['target'])], axis = 1)
pca_df.head()


# In[44]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pca_df['target'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
              , pca_df.loc[indicesToKeep, 'PC2']
              , c = color
              , s = 50)
ax.legend(targets)
ax.grid()


# In[45]:


pca.explained_variance_ratio_


# In[46]:


k_rng = range(1,15)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data_pca)
    sse.append(km.inertia_)
    
sse


# In[47]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)


# In[48]:


k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters = k)
    km_labels = km.fit_predict(data_pca)
    score = silhouette_score(data_pca, km_labels)
    print("For k =", k, ", the silhouette score is :", score,)


# In[49]:


model = KMeans(random_state=0) 
visualizer = KElbowVisualizer(model, k=(2,15), metric='silhouette', timings=False)

visualizer.fit(data_pca)    
visualizer.poof()


# In[50]:


km = KMeans(n_clusters = 4)
km


# In[51]:


y_predicted = km.fit_predict(data_pca)
y_predicted


# In[52]:


pca_df = pd.concat([data_pca, pd.DataFrame(y_predicted, columns = ['target'])], axis = 1)
pca_df.head()


# In[53]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0, 1, 2, 3]
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = pca_df['target'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
              , pca_df.loc[indicesToKeep, 'PC2']
              , c = color
              , s = 50)
ax.legend(targets)
ax.grid()


# In[54]:


linkage_data = linkage(scaled_data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.hlines(y=30,xmin=0,xmax=20000,lw=3,linestyles='--',color='k')
plt.show()


# In[55]:


hierarchical_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_predicted = hierarchical_cluster.fit_predict(scaled_data)
y_predicted


# In[56]:


df['cluster'] = y_predicted
df.head()


# In[57]:


h_rng = range(2,15)
for h in h_rng:
    hierarchical_cluster = AgglomerativeClustering(n_clusters=h, affinity='euclidean', linkage='ward')
    hc_labels = hierarchical_cluster.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, hc_labels)
    print("For k =", h, ", the silhouette score is :", score,)


# In[58]:


sns.heatmap(scaled_data.corr())


# In[59]:


pca = PCA(n_components = 2)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)

data_pca = pd.DataFrame(data_pca, columns = ['PC1', 'PC2'])
data_pca.head()


# In[60]:


sns.heatmap(data_pca.corr())


# In[61]:


linkage_data = linkage(data_pca, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.hlines(y=30,xmin=0,xmax=20000,lw=3,linestyles='--',color='k')
plt.show()


# In[62]:


hierarchical_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_predicted = hierarchical_cluster.fit_predict(data_pca)
y_predicted


# In[63]:


pca_df = pd.concat([data_pca, pd.DataFrame(y_predicted, columns = ['target'])], axis = 1)
pca_df.head()


# In[64]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0, 1, 2, 3]
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = pca_df['target'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
              , pca_df.loc[indicesToKeep, 'PC2']
              , c = color
              , s = 50)
ax.legend(targets)
ax.grid()


# In[65]:


pca.explained_variance_ratio_


# In[66]:


h_rng = range(2,15)
for h in h_rng:
    hierarchical_cluster = AgglomerativeClustering(n_clusters=h, affinity='euclidean', linkage='ward')
    hc_labels = hierarchical_cluster.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, hc_labels)
    print("For k =", h, ", the silhouette score is :", score,)


# In[ ]:




