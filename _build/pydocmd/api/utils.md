<h1 id="tensortools.utils">tensortools.utils</h1>


<h2 id="tensortools.utils.standardize">standardize</h2>

```python
standardize(arr)
```

Normalizes an array by subtracting the mean and dividing by the standard deviation. See
https://en.wikipedia.org/wiki/Feature_scaling.

>>> standardize([1, 2, 3])
array([-1.22474487,  0.        ,  1.22474487])

**Arguments**:

- `arr`: The array to normalize.

**Returns**:

The normalized array.

<h2 id="tensortools.utils.iou">iou</h2>

```python
iou(box_a, box_b)
```

Calculates the IOU between two boxes.

For example:

>>> iou([0.5, 0.5], [1, 1])
0.25

**Arguments**:

- `box_a`: 
- `box_b`: 

**Returns**:



<h2 id="tensortools.utils.avg_iou">avg_iou</h2>

```python
avg_iou(annotations, anchors)
```

Calculates the average iou between the given anchors. Only the max IOU between each annotation and anchor is used
to calculate the average.


**Arguments**:

- `annotations`: The annotations, shape [n_annotations, 2].
- `anchors`: The anchors, shape [n_anchors, 2].

**Returns**:

The average IOU.

<h2 id="tensortools.utils.count">count</h2>

```python
count(arr, value)
```

Counts the occurrences of the value in the given iterable.

**Arguments**:

- `arr`: The iterable.
- `value`: The value to count.

**Returns**:

The number of occurrences.

<h2 id="tensortools.utils.download_file">download_file</h2>

```python
download_file(url, dst)
```

Downloads a file from a url to the given destination with % finished bar.

**Arguments**:

- `url`: The url.
- `dst`: The destination.

