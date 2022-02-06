import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Reading the file from the folder
#NOTE: csv should be in the same folder as that of the program
coaches_career_df = pd.read_csv('.\data\coaches_career.csv')

#Cleaning the dataset
#Removing null values, replacing infinity with nan and then dropping nan values
#success index is defined by number of wins upon the total number of games played
coaches_career_df["season_success_index"] = coaches_career_df["season_win"]/(coaches_career_df["season_loss"]+coaches_career_df["season_win"])
coaches_career_df["playoff_success_index"] = coaches_career_df["playoff_win"]/(coaches_career_df["playoff_loss"]+coaches_career_df["playoff_win"])
coaches_career_df['season_success_index'] = coaches_career_df['season_success_index'].fillna(0)
coaches_career_df['playoff_success_index'] = coaches_career_df['playoff_success_index'].fillna(0)
coaches_career_df['playoff_success_index'] = coaches_career_df["playoff_success_index"].replace(np.inf, np.nan)
coaches_career_df = coaches_career_df.dropna()

#Visualization of Raw data
plt.figure(figsize=(10,4))
plt.bar(coaches_career_df['season_success_index'],coaches_career_df['playoff_success_index'], width=0.1)
plt.title('Bar Graph od season v/s playoff')
plt.xlabel('Season Success Index')
plt.ylabel('Playoff Success Index')
plt.show()

#Using K-means to identify the clusters
coach_success_df = coaches_career_df[['season_success_index', 'playoff_success_index']].copy()
fig = plt.figure(figsize=(14,14))
kmin, kmax = 2, 11
for k in range(kmin, kmax+1):
  kmeans = KMeans(k)
  kmeans.fit(coach_success_df)
  identified_clusters = kmeans.fit_predict(coach_success_df)
  coaches_career_clusters = coaches_career_df.copy()
  coaches_career_clusters['Clusters'] = identified_clusters 
  coaches_fig = fig.add_subplot(4,3,k-1, title= "K = "+ str(k))
  coaches_fig.scatter(coaches_career_clusters['season_success_index'], \
                      coaches_career_clusters['playoff_success_index'], \
                      c = coaches_career_clusters['Clusters'], cmap='rainbow')

"""From  the above analysis we can see that some teams have never played playoffs but have played many seasons
the light blue, orange, purple and lightgreen clusters indicate all such points
By looking at the files we can see that we have some redudant files which can be eliminated. 
Player_playoffs have season wise differentiation of the players where as player_playoffs_career has the entire data 
about a particular player which is cummulative of all seasons. 
Basically Player_playoffs_career is an aggregation of player_playoffs and thus can be derived.
"""

kmin = 2
kmax = 16


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points):
  sse = []
  inertia = []
  for k in range(kmin, kmax):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
    inertia.append(kmeans.inertia_)
  return sse,inertia

c, i = calculate_WSS(np.array(coach_success_df))
k_elbow = np.arange(kmin,kmax)
plt.figure(figsize=(8,5))
plt.title("The Elbow Method using euclidean distance")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Distance")
plt.plot(k_elbow, c, 'bx-', color ="blue")
plt.show()

plt.figure(figsize=(8,5))
plt.title("The Elbow Method using Inertia")
plt.xlabel("Values of K")
plt.ylabel("Inertia")
plt.plot(k_elbow, i, 'bx-', color ="blue")
plt.show()

"""The elbow method is not clearly showing the optimal k value. Thus we can say that, the data is not very clustered intrinsically."""

sil = []

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(kmin, kmax):
  kmeans = KMeans(n_clusters = k).fit(coach_success_df)
  labels = kmeans.labels_
  sil.append(silhouette_score(coach_success_df, labels, metric = 'euclidean'))

s = np.array(sil)
k = np.arange(kmin,kmax)
plt.title("Silhouette Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette score")
plt.plot(k, s, 'bx-', color ="blue")
plt.show()

"""K=2 is optimal."""

kmeans = KMeans(2)
kmeans.fit(coach_success_df)
identified_clusters = kmeans.fit_predict(coach_success_df)
coaches_career_clusters = coaches_career_df.copy()
coaches_career_clusters['Clusters'] = identified_clusters 
plt.title("For K = 2")
plt.scatter(coaches_career_clusters['season_success_index'],coaches_career_clusters['playoff_success_index'], c = coaches_career_clusters['Clusters'], cmap='rainbow')
plt.show()
