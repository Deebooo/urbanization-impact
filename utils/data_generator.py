import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, image_filenames, batch_size=12, dim=(512, 512), n_channels=3, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_filenames_temp = [self.image_filenames[k] for k in indexes]
        X, y = self.data_generation(image_filenames_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, image_filenames_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, filename in enumerate(image_filenames_temp):
            img_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename.replace('img', 'mask'))  # Adjust as per your naming convention

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Image not found at path: {img_path}")
            img = cv2.resize(img, self.dim)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found at path: {mask_path}")
            mask = cv2.resize(mask, self.dim)
    
            X[i] = img / 255.0
            y[i] = np.expand_dims(mask / 255.0, axis=-1)
    
        return X, y