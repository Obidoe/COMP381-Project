# COMP 381 ON1 - FINAL PROJECT: MACHINE LEARNING IMPLEMENTATION
# Zac Adams, Tegjot Dilawari

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import time
import os

os.makedirs('output', exist_ok=True)

# load dataset
cc_data = pd.read_csv('cc-data.csv')
#print(cc_data.head())

# select features
# we will use balance, purchases, oneoff_purchases, installments_purchases, credit_limit, and payments
useable_data = cc_data[['CREDIT_LIMIT', 'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                        'PAYMENTS']]
useable_data = useable_data.dropna()
#print(useable_data)

# 500 row sample of data
scaler = StandardScaler()
sample = useable_data.sample(n=500, random_state=1)
scaled_sample = scaler.fit_transform(sample)

# KMEANS

# KMEDOIDS

# HIERARCHICAL
def hierarchical(method, cut_height):
    start = time.time()

    plt.figure(figsize=(12, 6))
    z_link = hierarchy.linkage(scaled_sample, method=method)
    hierarchy.dendrogram(z_link, color_threshold=cut_height, truncate_mode='lastp', p=30)
    plt.title(f'{method.title()}-Link Hierarchical')
    plt.axhline(y=cut_height, color='r', linestyle='--', label=f'Cut Height = {cut_height}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/{method}_{cut_height}_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    clusters = fcluster(z_link, t=cut_height, criterion='distance')
    sample['cluster'] = clusters
    cluster_summary = sample.groupby('cluster').mean()
    cluster_summary = cluster_summary.sort_values(by='CREDIT_LIMIT')

    plt.figure(figsize=(12, 6))
    cluster_summary.T.plot(kind='bar', figsize=(12, 6))
    plt.title(f'{method.title()}-Link Cluster Feature Comparison: Cut Height: {cut_height}')
    plt.ylabel('Average Value')
    plt.xlabel('Features')
    plt.xticks(rotation=360, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'output/{method}_{cut_height}_cluster_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    end = time.time()

    print(f'{method}-Link Hierarchical (500 Sample):')
    print(f'Execution time: {end - start:.4f} seconds')
    print(f'\nCluster Summary: ')
    print(cluster_summary)


# RUN

# Kmeans test on 5 clusters
# Kmeans test on 6 clusters

# Kmedoids test on 5 clusters
# Kmedoids test on 6 clusters

hierarchical('single', 4)  # produces 6 clusters
hierarchical('complete', 4)  # produces 16 clusters

hierarchical('single', 10)  # produces 2 clusters
hierarchical('complete', 10)  # produces 5 clusters


