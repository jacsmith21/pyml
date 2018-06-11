<h1 id="tensortools.ops">tensortools.ops</h1>


<h2 id="tensortools.ops.fully_connected">fully_connected</h2>

```python
fully_connected(inputs, units, use_bias=True, activation=None, keep_prob=<tf.Tensor 'default_keep_prob:0' shape=() dtype=float32>, initializer=None)
```

A fully connected layer with dropout included.

**Arguments**:

- `inputs`: The inputs in the shape of [batch_size, previous_layer_units]
- `units`: The number of units for the layer.
- `use_bias`: Whether to use bias.
- `activation`: The TensorFlow activation function to use.
- `keep_prob`: The keep probability to use for the dropout layer.
- `initializer`: The kernal initializer.

**Returns**:



<h2 id="tensortools.ops.flatten">flatten</h2>

```python
flatten(tensor, has_batch=False)
```

Flattens a tensor.

**Arguments**:

- `tensor`: Tensor of shape [batch_size, dim1, dim2 ... dimn] or [dim1, dim2 ... dimn]
- `has_batch`: Whether or not the tensor has a batch dimension. If so, it is preserved.

**Returns**:

Tensor of shape [batch_size, dim1*dim2*...*dimn] or [dim1*dim2* ... *dimn]

