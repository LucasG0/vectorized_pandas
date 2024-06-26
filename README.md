# vectorized_pandas

`Series.apply` and `DataFrame.apply(axis=1)` tend to be overused by pandas users because of their simplicity, despite their limited performances.

`vectorized_pandas` is a Python package automatically replacing costly pandas `apply` calls by builtin/vectorized operations, if possible.

```python
>>> from vectorized_apply import replace_apply

>>> input_code = """
def func(row):
    if row["A"] == 1:
        if row["B"] == 1:
            return row["C"]
        else:
            return row["D"]
    else:
        if row["B"] == 2:
            return row["E"]
        else:
            return 0

df["H"] = df.apply(func, axis=1)
"""

>>> replace_apply(input_code)
"""
import numpy as np

df["H"] = np.select(
    conditions=[(df["A"] == 1) & (df["B"] == 1), df["A"] == 1, df["B"] == 2],
    choices=[df["C"], df["D"], df["E"]],
    default=0)
"""
```
