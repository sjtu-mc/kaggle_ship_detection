from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
import keras.backend as K


def iou_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (K.sum(intersection)+smooth) / (K.sum(y_true_f+y_pred_f) - K.sum(intersection) + smooth)
    return score

def bce_iou_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + 5*(1 - iou_coef(y_true, y_pred))

def show_loss(loss_history):
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['iou_coef'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_iou_coef'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('iou_coef')