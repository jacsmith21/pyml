<h1 id="tensortools.ssd">tensortools.ssd</h1>


<h2 id="tensortools.ssd.generate_anchors">generate_anchors</h2>

```python
generate_anchors(annotations, fm_sizes, n_clusters)
```

Generate anchors for an SSD model using KMeans. First, the algorithm matches each annotation with the appropriate
feature map. It does this by calculating the highest IOU between the annotation and one feature map cell. For
example, a feature map of [10, 10] and annotation of [0.1, 0.1] would have an IOU of 1. Next, it runs a KMeans
algorithm for each set of annotations.

**Arguments**:

- `annotations`: The annotations, shape `[n_samples, 4]`. Each annotation should be in the following form:
`[r1, c1, r2, c2]` where the first point is the ULC and the second point is the LRC. Each value should be in the
range `[0, 1]`
- `fm_sizes`: The feature map sizes, shape `[n_fm, 2]`. For example, `[[13, 13], [5, 5]]`
- `n_clusters`: The number of clusters (aka the number of desired bounding boxes for each feature map).

**Returns**:

The anchors boxes, shape `[n_fm, n_clusters, 2]`. Each value is in its normalized form in the range of
[0, 1].

