from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
import keras.backend as K
from segmentation_models.losses import jaccard_loss

def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_iou_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + 5*(1 - iou_coef(y_true, y_pred))

def show_loss(loss_history):
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['my_iou_metric'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_my_iou_metric'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('my_iou_metric')