<h1 id="tensortools.abc">tensortools.abc</h1>


<h2 id="tensortools.abc.abstract">abstract</h2>

```python
abstract(func)
```

An abstract decorator. Raises a NotImplementedError if called.

**Arguments**:

- `func`: The function.

**Returns**:

The wrapper function.

<h2 id="tensortools.abc.LazyLoader">LazyLoader</h2>

```python
LazyLoader(self, local_name, parent_module_globals, name)
```

Lazily import a module, mainly to avoid pulling in large dependencies.

`contrib`, and `ffmpeg` are examples of modules that are large and not always
needed, and this allows them to only be loaded when they are used.

