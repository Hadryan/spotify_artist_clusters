# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import dataset
dataset = pd.read_csv("data_by_artist.csv")
X = dataset.iloc[:,1:13].values

# Carry out feature scaling
sc = StandardScaler ()
X = sc.fit_transform(X)


# how many PCA components do we need?
pca = PCA(n_components = 12)
pca.fit(X)

variance = pca.explained_variance_ratio_
var = np.cumsum(np.round(variance,decimals=3)*100)

plt.ylabel('% Variance Explained')
plt.xlabel('No. of Features')
plt.ylim(20,110)
plt.xlim(0,13)
plt.plot(var)


# Fit transform X based on the number of components
pca = PCA(n_components = 8)
pca.fit_transform (X)

# Elbow method for KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    

# Training the KMeans Algorithm
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)


# Create a Category Column in the Dataset
dataset['Category'] = y_kmeans+1
dataset.head(10)


"""
We can use this by taking an example of an artist and recommending others in
the same category. In this case let us say that you are a huge fan of 
Frank Sinatra, which other artists are in the same cluster as Chairman of
the Board?"""

like_sinatra = dataset[dataset['Category']==2]


"""
Unfortunately this leaves us with 8,062 musicians. Perhaps this would be good
for a huge service such as Spotify. However, chances that there are 8,062
musicians with the qualities of Sinatra is slim.

Can we improve this? Perhaps not, clustering is not a good recommender. 
However, it can be used to divide the artists into various clusters based
on the the specific qualities of their music"""

musicians_by_category = dataset.iloc[:,[0,15]]
musicians_by_category.to_csv('musicians_by_category.csv', index = False, header = True)



    

