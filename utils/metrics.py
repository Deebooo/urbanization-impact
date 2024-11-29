import tensorflow as tf
from tensorflow.keras import backend as K

def jacard_coef(y_true, y_pred):
    """Calculate the Jaccard Coefficient between the true and predicted labels."""
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return jac