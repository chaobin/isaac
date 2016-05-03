# isaac
This is the Python files I put together during writing up a series of [blogs](http://chaobin.github.io) on machine learning.

# requirements

- numpy
- pandas (optional)
- matplotlib
- scipy

# examples

```python

>>> from isaac.models.regressions import LinearRegression
>>> from isaac.optimizers.gradient import Descent
>>> model = LinearRegression.from_dimension(10)
...
>>> X, Y = get_XY_from_frame(...)
>>> descent = Descent(model, X, Y, 0.001)
>>> print(model.cost(X, Y)

```
