import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

class COVID19Dataset(keras.utils.Sequence):
    def __init__(self, input_path, mask_path, image_size=(256, 256), batch_size=4):
        self.input_path = input_path
        self.mask_path = mask_path
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.mask_path) // self.batch_size
    
    def __getitem__(self, index):
        i = index * self.batch_size

        batch_input_paths = self.input_path[i: i + self.batch_size]
        batch_mask_paths = self.mask_path[i: i + self.batch_size]

        x = np.zeros((self.batch_size, ) + self.image_size + (1,), dtype='float32')
        y = np.zeros((self.batch_size, ) + self.image_size + (1,), dtype='float32')

        for j, (image_path, mask_path) in enumerate(zip(batch_input_paths, batch_mask_paths)):
            image = load_img(image_path, target_size=self.image_size, color_mode='grayscale')
            mask = load_img(mask_path, target_size=self.image_size, color_mode="grayscale")

            x[j] = np.expand_dims(image, 2) / 255.0
            y[j] = ((np.expand_dims(mask, 2) / 255.0) > 0.5) * 1

        return x, y
    
