import numpy as np

def my_roc_auc(classes : np.ndarray,
               predictions : np.ndarray,
               weights : np.ndarray = None) -> float:
    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1

    idx = np.argsort(predictions)

    predictions = predictions[idx]
    weights     = weights    [idx]
    classes     = classes    [idx]

    weights_0 = weights * (classes == 0)
    weights_1 = weights * (classes == 1)

    cumsum_0 = weights_0.cumsum()
    return (cumsum_0 * weights_1).sum() / (weights_1 * cumsum_0[-1]).sum()

