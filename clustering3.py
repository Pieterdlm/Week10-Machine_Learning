import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv("clusteringtabelsegment.csv")

label_encoder = LabelEncoder()
data['SEGMENT_NAME'] = label_encoder.fit_transform(data['SEGMENT_NAME'])
data['COUNTRY_EN'] = label_encoder.fit_transform(data['COUNTRY_EN'])

X = data[['SEGMENT_NAME', 'COUNTRY_EN']].values

scaler = MinMaxScaler(feature_range=(0, 100))
X_scaled = scaler.fit_transform(X)

clustering_model = KMeans(n_clusters=8)

clustering_model.fit(X_scaled)

labels = clustering_model.labels_
data['cluster_label'] = labels

print(data[['RETAILER_NAME', 'cluster_label', 'SEGMENT_NAME', 'COUNTRY_EN']])

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(clustering_model.cluster_centers_[:, 0], clustering_model.cluster_centers_[:, 1], c='red', marker='x', s=100)
plt.xlabel('SEGMENT_NAME')
plt.ylabel('COUNTRY_EN')
plt.title('Clustering Results')
plt.show()