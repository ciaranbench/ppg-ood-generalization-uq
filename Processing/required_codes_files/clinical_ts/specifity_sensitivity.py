## the code is written by oscar Pfeffer from PTB








__all__ = ['specificity','sensitivity','general_threshold', 'recall_score_threshold', 'eval_sensitivity_specificity']


import sklearn
import numpy as np
import warnings
import typing

def specificity(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute the specificity (or true negative rate). Specificity is
    also known as the recall score of the negative class.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.

    Returns
    -------
    float
        Specificity score.

    See Also
    --------
    sensitivity
    """
    return sklearn.metrics.recall_score(target, prediction, pos_label=0)


def sensitivity(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute the sensitivity (or true positive rate). Sensitivity is
    also known as the recall score of the positive class.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.

    Returns
    -------
    float
        Sensitivity score.

    See Also
    --------
    specificity
    """
    return sklearn.metrics.recall_score(target, prediction, pos_label=1)


def general_threshold(
    target: np.ndarray,
    prediction: np.ndarray,
    metric: typing.Callable[[np.ndarray, np.ndarray], float],
    metric_value: float,
    greater_than: bool = True,
) -> float:
    """
    Find the threshold that sets the given metric closest to `metric_value`.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    metric : Callable[[np.ndarray, np.ndarray], float]
        A metric function that takes target and prediction arrays as input.
    metric_value : float
        The value of the metric to be achieved.
    greater_than : bool
        True if the metric is supposed to be higher than `metric_value`,
        false otherwise.

    Returns
    -------
    float
        The threshold that achieves the metric.
    """
    thresholds = np.unique(prediction)
    if metric(target, prediction > thresholds[0]) > metric(
        target, prediction > thresholds[-1]
    ):
        if greater_than:
            thresholds = thresholds[::-1]
    else:
        if not (greater_than):
            thresholds = thresholds[::-1]
    for threshold in thresholds:
        prediction_binary = prediction > threshold
        res = metric(target, prediction_binary)
        if greater_than and res > metric_value:
            return threshold
        elif not (greater_than) and res < metric_value:
            return threshold
    warnings.warn("Could not find threshold achieving target metric_value.")
    return 0.0


def recall_score_threshold(
    target: np.ndarray,
    prediction: np.ndarray,
    recall_value: float,
    pos_label: 1 | 0 = 1,
    greater_than: bool = True,
    dtype: np.dtype = np.float32,
) -> float:
    """
    Compute the classification threshold so that the recall score is
    closest to the specified value, but greater (or lower).

    Default: The threshold is computed for the sensitivity score.
    Set sensitivity=False for specificity.

    The threshold is set as the next floating point number after
    (before) the value of the prediction that needs to be classified
    positive (negative) to achieve the desired recall score.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    recall_value : float
        The desired recall score.
    pos_label : 1 | 0, optional
        1 to compute the threshold for sensitivity, 0 for
        specificity.
    greater_than : bool, optional
        True to let the recall score be higher than recall_value, false
        to let the recall score be lower than recall_value.
    dtype : np.dtype, optional
        The data type of the threshold, by default np.float32

    Returns
    -------
    float
        The classification threshold.
    """

    if np.any(prediction < 0.0) or np.any(prediction > 1.0):
        raise ValueError("All values in prediction must be between 0 and 1.")

    if pos_label:
        label_prediction = np.sort(prediction[target == pos_label])[::-1]
    else:
        label_prediction = np.sort(prediction[target == pos_label])

    if label_prediction.size == 0:
        warnings.warn(
            "Recall is ill-defined and the threshold is set to 0 due to no true samples."
        )
        return 0

    true_label = recall_value * len(label_prediction)
    threshold_index = max(0, np.ceil(true_label - 1).astype(int))

    # set the threshold slightly over/under the prediction value
    # to avoid >= or > issues while computing binary predictions
    # np.mod(x+y,2) is the same as xor(x,y) for two binary values
    threshold = np.nextafter(
        dtype(label_prediction[threshold_index]),
        np.mod(pos_label + greater_than, 2, dtype=dtype),
    )
    return threshold


def eval_sensitivity_specificity(
    target: np.ndarray,
    prediction: np.ndarray,
    recall_value: float,
    pos_label: 1 | 0 = 1,
    greater_than: bool = True,
) -> tuple[float, float]:
    """Evaluate the sensitivity and specificity of a given prediction by fixing
    either the sensitivity or the specificity to the specified recall_value.

    Parameters
    ----------
    target : np.ndarray
        Ground truth values.
    prediction : np.ndarray
        Model output predictions.
    recall_value : float
        The value of the recall score to be achieved.
    pos_label : Callable[[np.ndarray, np.ndarray], float]
        1 to fix the sensitivity to recall_value, 0 to fix the specificity.
    greater_than : bool
        True if the recall score is supposed to be higher than `recall_value`,
        false otherwise.

    Returns
    -------
    Tuple[float, float]
        The sensitivity and specificity of the prediction.
    """
    threshold = recall_score_threshold(
        target,
        prediction,
        recall_value=recall_value,
        pos_label=pos_label,
        greater_than=greater_than,
    )

    prediction_binary = prediction > threshold

    sens = sensitivity(target, prediction_binary)
    spec = specificity(target, prediction_binary)

    return sens, spec