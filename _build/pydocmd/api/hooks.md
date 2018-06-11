<h1 id="tensortools.hooks">tensortools.hooks</h1>


<h2 id="tensortools.hooks.IntervalHook">IntervalHook</h2>

```python
IntervalHook(self, interval)
```

A hook which runs every # of iterations. Useful for subclassing.

<h3 id="tensortools.hooks.IntervalHook.session_run_args">session_run_args</h3>

```python
IntervalHook.session_run_args(self, run_context)
```

Create the session run arguments.

**Arguments**:

- `run_context`: The run context.

**Returns**:

The list of arguments to run.

<h2 id="tensortools.hooks.GlobalStepIncrementor">GlobalStepIncrementor</h2>

```python
GlobalStepIncrementor(self)
```

Increments the global step after each `Session` `run` call. Useful for models which do not use optimizers.

