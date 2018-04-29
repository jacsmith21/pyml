[![Build Status](https://travis-ci.org/jacsmith21/tensortools.png?branch=master)](https://travis-ci.org/jacsmith21/tensortools)
[![Docs](https://readthedocs.org/projects/tensortools/badge/?version=latest)](https://tensortools.readthedocs.io/)

# TensorTools
A simple repository which implements common things I use in my ML projects. See `examples/` for some examples :sunglasses:. 

Better API documentation coming ASAP!

# Installation
```bash
pip install tensortools
```

# Examples
```python
import tensorflow as tf
import tensortools as tt
import numpy as np

# Generate images with random shapes for object detection
image_generator = tf.generator.Generator()
images = image_generator.generate(n_images=1, height=100, width=100, max_shapes=2)
image = images[0]
tt.image.show(image)

# Some simple ops
vector = tt.ops.flatten(image, has_batch=False)
with tf.Session() as sess:
  vector = sess.run([vector])

# Generate anchors for YOLO algorithm using KMeans
annotations = load_annotations()
anchors = tt.yolo.generate_anchors()
```
