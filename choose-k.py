import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def cluster(faces, max_k):
    data = [np.array(face["features"]) for face in faces]

    # Normalise the data in each column so as not to favour one over another
    data = data - np.min(data, axis=0)
    data = data / np.max(data, axis=0)

    cost = np.zeros(max_k)
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k)
        model = kmeans.fit(data)
        cost[k] = model.inertia_

    print(cost)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, max_k), cost[2:max_k])
    ax.set_xlabel('k')
    ax.set_xticks(np.arange(0, max_k + 1, 1))
    ax.set_ylabel('cost')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face clustering, cluster creator')
    parser.add_argument('--input', help='Input file of JSON data', default='face_data.json')
    parser.add_argument('--max-k', help='Highest K you want to try (lowest is 2)', default=20)
    args = parser.parse_args()

    with open(args.input) as json_file:
        data = json.load(json_file)

    cluster(data, args.max_k)
