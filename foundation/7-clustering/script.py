
import pandas as pd
votes = pd.read_csv("114_congress.csv")


print(votes["party"].value_counts())
print(votes.mean())


#Compute the Euclidean distance between the first row and the third row.
from sklearn.metrics.pairwise import euclidean_distances

print(euclidean_distances(votes.iloc[0,3:].reshape(1, -1), votes.iloc[1,3:].reshape(1, -1)))
distance = euclidean_distances(votes.iloc[0,3:].reshape(1, -1), votes.iloc[2,3:].reshape(1, -1))



import pandas as pd
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=1)
senator_distances = kmeans_model.fit_transform(votes.iloc[:, 3:])


labels = kmeans_model.labels_
print(pd.crosstab(labels, votes["party"]))


democratic_outliers = votes[(labels == 1) & (votes["party"] == "D")]
print(democratic_outliers)

import matplotlib.pyplot as plt
plt.scatter(x=senator_distances[:,0], y=senator_distances[:,1], c=labels)
plt.show()



# finding the most extreme 
extremism = (senator_distances ** 3).sum(axis=1)
votes["extremism"] = extremism
votes.sort_values("extremism", inplace=True, ascending=False)
print(votes.head(10))