import yaml
import tensorflow as tf
import os
import random
import time
from unet import build_unet
from losses import LossConfig
from metrics import *
from dataloader import COVID19Dataset

def test():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    hp = config['hyperparameters']
    paths = config['paths']
    loss_type = hp['loss_function']
    
    test_img_dir = paths['test_x']
    test_mask_dir = paths['test_y']

    all_imgs = sorted([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)])
    all_masks = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir)])

    combined = list(zip(all_imgs, all_masks))
    random.seed(1332)
    random.shuffle(combined)
    
    selected_data = combined[:944]
    test_imgs, test_masks = zip(*selected_data)

    test_generator = COVID19Dataset(
        list(test_imgs), 
        list(test_masks), 
        batch_size=hp['batch_size'], 
        image_size=tuple(hp['input_size'])
    )

    model = build_unet(tuple(hp['input_shape']), hp['num_classes'])
    weights_path = "dice_focal_loss_a0.45_g1.25_best.keras"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f">> Ağırlıklar '{weights_path}' dosyasından yüklendi.")
    else:
        print(f">> Hata: {weights_path} bulunamadı!")
        return

    if loss_type == "binary_crossentropy":
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_fn_class = LossConfig.loss[loss_type]
        if loss_type in LossConfig.alpha:
            if loss_type in LossConfig.gamma:
                loss_fn = loss_fn_class(alpha=hp['alpha_range'][0], gamma=hp['gamma_range'][0])
            else:
                loss_fn = loss_fn_class(alpha=hp['alpha_range'][0])
        else:
            loss_fn = loss_fn_class()

    METRICS = [
        dice_score, jaccard_score, accuracy_score, 
        precision_score, recall_score, specificity_score, f1_score,
        tf.keras.metrics.AUC(name='auc_score')
    ]

    model.compile(optimizer='adam', loss=loss_fn, metrics=METRICS)

    start_inference = time.time()
    results = model.evaluate(test_generator, verbose=1)
    end_inference = time.time()

    total_time = end_inference - start_inference
    time_per_image = total_time / 944

    print("\n" + "="*30)
    print(f"Inference Time: {time_per_image:.6f} second per image")
    print("="*30)
    
    for name, value in zip(model.metrics_names, results):
        print(f"{name:20}: {value:.4f}")

if __name__ == "__main__":
    test()