#!/usr/bin/env python
# coding: utf-8

# In[50]:


from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data_s1 = np.genfromtxt('./datasets/s/s1.txt',delimiter='    ',dtype=int)


# In[42]:


data_s1.shape


# In[15]:


fig = plt.figure()
plt.scatter(data_s1[:, 0], data_s1[:, 1])


# In[39]:


data_s1_gt = np.genfromtxt('./datasets/s/s1-cb.txt',delimiter=' ',dtype=int)


# In[59]:


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(data_s1, quantile=0.05, n_samples=2500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data_s1)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# In[66]:


fig = plt.figure()
plt.scatter(data_s1[:, 0], data_s1[:, 1], c = labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c = 'red', alpha=0.5)
plt.scatter(data_s1_gt[:, 0], data_s1_gt[:, 1], c = 'blue', alpha=0.5)


# In[67]:


data_g2 = np.genfromtxt('./datasets/g2/g2-2-10.txt',delimiter='    ',dtype=int)


# In[68]:


data_g2.shape


# In[69]:


data_g2


# In[70]:


fig = plt.figure()
plt.scatter(data_g2[:, 0], data_g2[:, 1])


# In[77]:


data_g2_gt = np.genfromtxt('./datasets/g2/g2-2-10-gt.txt',delimiter=' ',dtype=int)


# In[78]:


data_g2_gt


# In[73]:


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(data_g2, quantile=0.25, n_samples=2500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data_g2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# In[79]:


fig = plt.figure()
plt.scatter(data_g2[:, 0], data_g2[:, 1], c = labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c = 'red', alpha=0.5)
plt.scatter(data_g2_gt[:, 0], data_g2_gt[:, 1], c = 'blue', alpha=0.5)


# In[ ]:




