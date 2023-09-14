## Load libraries

from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys    ## no need to install package


## Set working directory
os.chdir(r"C:\Users\nuran\PycharmProjects\CIS9660\Project")


## Import Data
mydata = pd.read_csv('vehicle_collisions_clustering2.csv')

#Remove outliers
def remove_outliers(df, z_thresh):
 z_scores = np.abs((df - df.mean()) / df.std())
 df = df[(z_scores < z_thresh).all(axis=1)]
 return df

x = remove_outliers(mydata, 3.0)


# Normalize the data using StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(x)
x = pd.DataFrame(normalized_data, columns=mydata.columns)



##Clustering
kmean=KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
kmean.fit(x)
kmean.cluster_centers_
labels=kmean.labels_



# Calculate the withinness-cluster SSE (sum of squared errors)
within_cluster_sse = kmean.inertia_

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(x, labels)



# Save the results to an Excel file
writer = pd.ExcelWriter('clustering_results.xlsx', engine='xlsxwriter')
x['labels'] = labels
x.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()

#Print summary statistics of the results
pd.set_option('display.max_columns', 500)
file = open('clustering_output.txt','wt')
sys.stdout = file
summary = x.groupby('labels').agg(['mean', 'std', 'count'])
print(summary)
print("Within-cluster SSE:", within_cluster_sse)
print("Silhouette Score:", silhouette_avg)


# make as panda dataframe for easy understanding
wcss = []
for i in range(1,10):
 kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
 kmeans.fit(x)
 wcss.append(kmeans.inertia_)
 print('Cluster', i, 'Inertia', kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('within-cluster_SSE') ##WCSS stands for total within-cluster sum of square
plt.show()
