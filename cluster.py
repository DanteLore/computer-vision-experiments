import argparse
import json

import numpy as np
from sklearn.cluster import KMeans


def cluster(faces, cluster_count):

    data = [np.array(face["features"]) for face in faces]

    # Normalise the data in each column so as not to favour one over another
    data = data - np.min(data, axis=0)
    data = data / np.max(data, axis=0)

    for face in faces:
        print(face["features"])

    # Get the face data
    data = [np.array(face["features"]) for face in faces]

    # Build the model using the face data
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans = kmeans.fit(data)

    # Get cluster numbers for each face
    labels = kmeans.predict(data)
    for (label, face) in zip(labels, faces):
        face["group"] = int(label)

    # Centroid values
    centroids = kmeans.cluster_centers_

    for (label, face) in zip(labels, faces):
        face["group"] = int(label)
        face["size"] = 40000

    for i in range(0, cluster_count):
        cluster = {
            "name": "group {0}".format(i),
            "filename": "cool.png",
            "size": 40000,
            "children": [c for c in faces if c["group"] == i]
        }
        yield cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face clustering, cluster creator')
    parser.add_argument('--input', help='Input file of JSON data', default='face_data.json')
    parser.add_argument('--output', help='Output file for clustered JSON data', default='clustered_data.json')
    parser.add_argument('--count', help='Number of clusters', default=4)
    args = parser.parse_args()

    with open(args.input) as json_file:
        data = json.load(json_file)

    graph = {
        "name": "Faces",
        "filename": "cool.png",
        "children": list(cluster(data, args.count))
    }

    with open(args.output, 'w') as outfile:
        json.dump(graph, outfile, indent=2, sort_keys=False)
