import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import time
import numpy as np

from losses import *
from metrics import *
from unet import build_unet
from plotting import plot_history
from preprocessing import load_data

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/estu/anaconda3/envs/new_tf/'
os.environ['LD_LIBRARY_PATH'] = '/home/estu/anaconda3/envs/new_tf/lib/:' + os.environ.get('LD_LIBRARY_PATH', '')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

METRICS = [
    dice_score, jaccard_score, accuracy_score, 
    precision_score, recall_score, specificity_score, f1_score,
    tf.keras.metrics.AUC(name='auc_score')
]

def train(train_data, loss, hyperparameters, val_data=None, output='output', callback=None):
    tf.keras.backend.clear_session()
    model = build_unet(tuple(hyperparameters['input_shape']), hyperparameters['num_classes'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
        loss=loss, 
        metrics=METRICS
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_score', 
        patience=10, 
        mode='max', 
        restore_best_weights=True
    )
    
    callbacks_list = [early_stop]
    if callback is not None:
        callbacks_list.append(callback)
    
    start_time = time.time()
    history = model.fit(
        train_data,
        epochs=hyperparameters['epochs'], 
        batch_size=hyperparameters['batch_size'],
        validation_data=val_data,
        callbacks=callbacks_list,
        verbose=1
    )
    end_time = time.time()
    
    avg_time_per_epoch = (end_time - start_time) / len(history.history['loss'])
    print(f"\nTraining Time: {avg_time_per_epoch:.4f} second per epoch")
    
    plot_history(history, output)
    return max(history.history['val_dice_score'])

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())

    paths = config['paths']
    hp = config['hyperparameters']
    loss_type = hp['loss_function']

    train_data, validation_data = load_data(paths, 0.75, hp['input_size'], hp['batch_size'])
    loss_fn_base = LossConfig.loss[loss_type]
    
    results_summary = []

    if loss_type in LossConfig.alpha:
        for alpha in hp['alpha_range']:
            if loss_type in LossConfig.gamma: 
                for gamma in hp['gamma_range']:
                    param_str = f"a{alpha}_g{gamma}"
                    print(f"\n--- EĞİTİM BAŞLIYOR: {loss_type} ({param_str}) ---")
                    
                    weight_name = f"{loss_type}_{param_str}_best.keras"
                    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        weight_name, monitor="val_dice_score", save_best_only=True, save_weights_only=True, mode="max"
                    )
                    
                    best_val_dice = train(train_data, loss_fn_base(alpha, gamma), hp, validation_data, output=f"{loss_type}_{param_str}", callback=checkpoint)
                    results_summary.append((param_str, best_val_dice))
            else:
                param_str = f"a{alpha}"
                print(f"\n--- EĞİTİM BAŞLIYOR: {loss_type} ({param_str}) ---")
                
                weight_name = f"{loss_type}_{param_str}_best.keras"
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    weight_name, monitor="val_dice_score", save_best_only=True, save_weights_only=True, mode="max"
                )
                
                best_val_dice = train(train_data, loss_fn_base(alpha), hp, validation_data, output=f"{loss_type}_{param_str}", callback=checkpoint)
                results_summary.append((param_str, best_val_dice))
    else:
        print(f"\n--- EĞİTİM BAŞLIYOR: {loss_type} ---")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{loss_type}_best.keras", monitor="val_dice_score", save_best_only=True, save_weights_only=True, mode="max")
        train(train_data, loss_fn_base, hp, validation_data, output=f"{loss_type}_results", callback=checkpoint)

    if results_summary:
        print("\n" + "="*30)
        print(" PARAMETRE TARAMA SONUÇLARI ")
        print("="*30)
        for params, score in results_summary:
            print(f"Parametreler: {params} | En İyi Val Dice: {score:.4f}")
        print("="*30)