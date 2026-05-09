import os
import random
from dataloader import COVID19Dataset

def load_data(paths, split_size, input_size, batch_size=4):
    train_img_list = sorted(list(map(lambda x: os.path.join(paths['train_x'], x), os.listdir(paths['train_x']))))
    train_masks_list =  sorted(list(map(lambda x: os.path.join(paths['train_y'], x), os.listdir(paths['train_y']))))


    random.Random(1332).shuffle(train_img_list)
    random.Random(1332).shuffle(train_masks_list)

    train_split = int(len(train_img_list) * split_size)

    train_imgs = train_img_list[:train_split]
    train_masks = train_masks_list[:train_split]

    val_imgs = train_img_list[train_split:]
    val_masks = train_masks_list[train_split:]

    train_data = COVID19Dataset(train_imgs, train_masks, batch_size=batch_size, image_size=tuple(input_size))
    validation_data = COVID19Dataset(val_imgs, val_masks, batch_size=batch_size, image_size=tuple(input_size))

    return train_data, validation_data