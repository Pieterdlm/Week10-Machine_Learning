import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("clusteringtabelv2.csv")

label_encoder = LabelEncoder()
data['PRODUCT_TYPE_EN'] = label_encoder.fit_transform(data['PRODUCT_TYPE_EN'])
data['PRODUCT_LINE_EN'] = label_encoder.fit_transform(data['PRODUCT_LINE_EN'])

X = data[['PRODUCT_TYPE_EN', 'PRODUCT_LINE_EN']].values

clustering_model = KMeans(n_clusters=8) 

clustering_model.fit(X)

labels = clustering_model.labels_

data['cluster_label'] = labels

print(data[['RETAILER_NAME', 'cluster_label', 'PRODUCT_TYPE_EN', 'PRODUCT_LINE_EN']])

print('====================================')
data['count'] = 1

cluster_count = data.groupby(['RETAILER_NAME', 'cluster_label']).agg({'PRODUCT_TYPE_EN': 'first',
                                                                     'PRODUCT_LINE_EN': 'first',
                                                                     'count': 'size'}).reset_index()

print(cluster_count)

max_count_index = cluster_count.groupby('RETAILER_NAME')['count'].idxmax()

result = cluster_count.loc[max_count_index]

print(result)

unique_cluster_labels = result['cluster_label'].unique()

colors = plt.cm.get_cmap('tab10', len(unique_cluster_labels))
plt.figure(figsize=(10, 6))
for i, cluster_label in enumerate(unique_cluster_labels):
    cluster_data = result[result['cluster_label'] == cluster_label]
    plt.scatter(cluster_data['PRODUCT_TYPE_EN'], cluster_data['PRODUCT_LINE_EN'], color=colors(i),
                label=f'Cluster {cluster_label}')
    for _, retailer in cluster_data.iterrows():
        plt.annotate(retailer['RETAILER_NAME'], (retailer['PRODUCT_TYPE_EN'], retailer['PRODUCT_LINE_EN']),
                     textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel('PRODUCT_TYPE_EN')
plt.ylabel('PRODUCT_LINE_EN')
plt.title('Clusters of Retailers')
plt.legend()
plt.show()  