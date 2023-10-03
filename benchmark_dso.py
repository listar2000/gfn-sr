from benchmark import NGUYEN_TESTS
from dso import DeepSymbolicRegressor
import numpy as np


if __name__ == '__main__':
    model = DeepSymbolicRegressor()
    nguyen = NGUYEN_TESTS[0]
    xs, ys, _ = nguyen()
    # Fit the model
    model.fit(xs.numpy(), ys.numpy())  # Should solve in ~10 seconds

    # View the best expression
    print(model.program_.pretty())
