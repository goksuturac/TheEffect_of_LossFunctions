from tensorflow.keras import backend as K
import tensorflow as tf


def confusion_matrix(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    tp = K.sum(y_true * y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    return tp, tn, fp, fn
    
def dice_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def jaccard_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())    

def accuracy_score(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)

def precision_score(y_true, y_pred):
    tp, _, fp, _ = confusion_matrix(y_true, y_pred)
    return (tp + K.epsilon()) / (tp + fp + K.epsilon())

def recall_score(y_true, y_pred):
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    return (tp + K.epsilon()) / (tp + fn + K.epsilon())

def specificity_score(y_true, y_pred):
    _, tn, fp, _ = confusion_matrix(y_true, y_pred)    
    return (tn + K.epsilon()) / (tn + fp + K.epsilon())

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (2 * precision * recall) / (precision + recall)

auc_instance = tf.keras.metrics.AUC()

def auc_score(y_true, y_pred):
    auc_instance.update_state(y_true, y_pred)
    return auc_instance.result()