<h1 id="tensortools.yolo">tensortools.yolo</h1>


<h2 id="tensortools.yolo.generate_anchors">generate_anchors</h2>

```python
generate_anchors(annotations, num_clusters)
```

Generate anchors for YOLO V2 & V3 using the algorithm described in the paper (KMeans).

**Arguments**:

- `annotations`: The annotations, shape `[n_annotations, 2]`.
- `num_clusters`: The amount of clusters to create.

**Returns**:

The centroids (annotations) that best cluster the given annotations, shape `[num_clusters, 2]`.

