import numpy as np


class ActivationFunc:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc
    
    def __call__(self, x):
        return self.func(x)
    
    def derivative(self, x):
        return self.dfunc(x)


class LossFunc:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc
    
    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred)
    
    def derivative(self, y_true, y_pred):
        return self.dfunc(y_true, y_pred)


def tail_diag(a):
    return a[..., None] * np.eye(a.shape[-1])


sigmoid_func = lambda x: 1 / (1 + np.exp(-x))
sigmoid = ActivationFunc(
    sigmoid_func,
    lambda x: tail_diag(sigmoid_func(x) * (1 - sigmoid_func(x)))
)

relu = ActivationFunc(
    lambda x: x * (x > 0),
    lambda x: tail_diag(x > 0)
)

def softmax_func(x):
    w = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return w / np.sum(w, axis=-1, keepdims=True)

softmax = ActivationFunc(
    softmax_func,
    lambda x: tail_diag(softmax_func(x)) - np.einsum("...i,...j->...ij", softmax_func(x), softmax_func(x))
)

cross_entropy = LossFunc(
    lambda y_true, y_pred: -np.sum(y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))),
    lambda y_true, y_pred: -y_true / np.clip(y_pred, 1e-7, 1 - 1e-7)
)

mse = LossFunc(
    lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2),
    lambda y_true, y_pred: -2 * (y_true - y_pred)
)

def labels_to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes), dtype=float)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None):
            self.iterable = iterable
            self.total = total
        
        def __iter__(self):
            return iter(self.iterable)
        
        def update(self, n):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

try:
    from itertools import batched
except ImportError:
    def batched(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]
