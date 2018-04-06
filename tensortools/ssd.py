def generate_anchors(annotations, num_clusters):
    annotations = np.array(annotations)

    indices = [random.randrange(len(annotations)) for _ in range(num_clusters)]
    initial_centroids = annotations[indices]
    return _k_means(annotations, initial_centroids)
