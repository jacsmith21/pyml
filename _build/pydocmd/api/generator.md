<h1 id="tensortools.generator">tensortools.generator</h1>


<h2 id="tensortools.generator.RectangleGenerator">RectangleGenerator</h2>

```python
RectangleGenerator(self, /, *args, **kwargs)
```

Generates rectangles.

<h2 id="tensortools.generator.CircleGenerator">CircleGenerator</h2>

```python
CircleGenerator(self, /, *args, **kwargs)
```

Generates circles.

<h2 id="tensortools.generator.TriangleGenerator">TriangleGenerator</h2>

```python
TriangleGenerator(self, /, *args, **kwargs)
```

Generates triangles.

<h2 id="tensortools.generator.Generator">Generator</h2>

```python
Generator(self, /, *args, **kwargs)
```

<h3 id="tensortools.generator.Generator.generate">generate</h3>

```python
Generator.generate(self, n_images, height, width, max_shapes, min_shapes=1, min_size=2, max_size=None, shape=None, allow_overlap=False)
```

Generates a fake object detection dataset of squares, triangles & circles!

**Arguments**:

- `n_images`: The amount images to generate.
- `height`: The height of the desired images.
- `width`: The width of the desired images.
- `max_shapes`: The max amount of shapes per image.
- `min_shapes`: The min amount of shapes per image.
- `min_size`: The min size of the shapes.
- `max_size`: The max size of the shapes.
- `shape`: The type of shape. If None, a shape is randomly chosen each time.
- `allow_overlap`: Whether or not to allow overlap.

**Returns**:


images: The generates images, shape [n_images, height, width, 3].

