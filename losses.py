import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K


class BinaryCrossentropyLoss(Loss):
    def call(self, y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

class DiceLoss(Loss):
    def dice_coefficient(self, y_true, y_pred):
            y_true = K.flatten(y_true)
            y_pred = K.flatten(y_pred)
            intersection = K.sum(y_true * y_pred)
            return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

    def call(self, y_true, y_pred):
         return 1 - self.dice_coefficient(y_true, y_pred)    


class WeightedBCELoss(Loss):
    def __init__(self, name = "weighted_bce_loss", **kwargs):
        super().__init__(name = name , **kwargs)
    def call(self, y_true, y_pred):
        y_true= tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0)-K.epsilon()


        w0 = tf.reduce_sum(tf.cast(tf.equal(y_true, 0), tf.float32)) + K.epsilon()
        w1 = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32)) + K.epsilon()

        loss = -((1/w0)* (1- y_true)*K.log(1-y_pred) + (1/w1)*y_true*K.log(y_pred))

        return tf.reduce_sum(loss)


class FocalLoss(Loss):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0 + K.epsilon()))


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45):
        super().__init__()
        self.alpha = alpha
    
    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        return (true_pos + self.alpha)/(true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + K.epsilon())
    
    def call(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)
    
class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.3, gamma=1.25):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        return (true_pos + self.alpha + K.epsilon()) / (true_pos + self.alpha*false_neg + (1-self.alpha)*false_pos + K.epsilon())
    
    def call(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        return K.pow((1-pt_1), self.gamma)
    
class DiceFocalLoss(Loss):
    def __init__(self, alpha=0.45, gamma=1.25):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)

    def call(self, y_true, y_pred):
        return self.dice_loss(y_true, y_pred), self.focal_loss(y_true, y_pred)   
    
class DiceWBCELoss(Loss):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.wbce_loss = WeightedBCELoss()
    
    def call(self, y_true, y_pred):
        return K.mean(self.dice_loss(y_true, y_pred) + self.wbce_loss(y_true, y_pred))
    
    

class LossConfig:
    normal = ["binary_crossentropy", "dice_loss", "weighted_bce", "dice_weighted_bce_loss", "jaccard_loss"]
    alpha = ["focal_loss", "tversky_loss", "focal_tversky_loss", "dice_focal_loss"]
    gamma = ["focal_loss", "focal_tversky_loss", "dice_focal_loss"]

    loss = {
        "binary_crossentropy": BinaryCrossentropyLoss,
        "dice_loss": DiceLoss,
        "weighted_bce": WeightedBCELoss,
        "focal_loss": FocalLoss,
        "tversky_loss": TverskyLoss,
        "focal_tversky_loss": FocalTverskyLoss,
        "dice_focal_loss": DiceFocalLoss,
        "dice_weighted_bce_loss": DiceWBCELoss
    }