import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from utils.gpu_setup import setup_gpus
from utils.metrics import jacard_coef
import tensorflow as tf

# Configure GPU settings
setup_gpus(memory_limit=17000)

# Load model
model_path = r"D:\Deep_learning_final\MODEL_7.0.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'jacard_coef': jacard_coef})

# TileGenerator class
class TileGenerator:
    def __init__(self, image_path, tile_size=(512, 512), batch_size=32):
        self.image_path = image_path
        self.tile_size = tile_size
        self.batch_size = batch_size

        with rasterio.open(self.image_path) as src:
            self.width = src.width
            self.height = src.height
            self.num_tiles_x = int(np.ceil(self.width / self.tile_size[0]))
            self.num_tiles_y = int(np.ceil(self.height / self.tile_size[1]))
            self.num_tiles = self.num_tiles_x * self.num_tiles_y

    def __len__(self):
        return int(np.ceil(self.num_tiles / self.batch_size))

    def __getitem__(self, index):
        batch_tiles = []

        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_tiles)

        with rasterio.open(self.image_path) as src:
            for i in range(start_index, end_index):
                row = i // self.num_tiles_x
                col = i % self.num_tiles_x

                window = Window(
                    col * self.tile_size[0],
                    row * self.tile_size[1],
                    self.tile_size[0],
                    self.tile_size[1]
                )

                tile = src.read(window=window)
                tile = tile.transpose(1, 2, 0)  # From (bands, rows, cols) to (rows, cols, bands)

                # Handle padding if tile is smaller than tile_size
                if tile.shape[0] != self.tile_size[1] or tile.shape[1] != self.tile_size[0]:
                    pad_width = (
                        (0, self.tile_size[1] - tile.shape[0]),
                        (0, self.tile_size[0] - tile.shape[1]),
                        (0, 0)
                    )
                    tile = np.pad(tile, pad_width, mode='constant', constant_values=0)

                # Normalize the tile
                tile = tile.astype(np.float32) / 255.0

                batch_tiles.append(tile)

        batch_tiles = np.stack(batch_tiles, axis=0)
        return batch_tiles

def merge_predictions(predictions, width, height, tile_size, output_path, meta, threshold=0.5):
    num_tiles_x = int(np.ceil(width / tile_size[0]))
    num_tiles_y = int(np.ceil(height / tile_size[1]))

    # Initialize empty array for the full prediction
    full_prediction = np.zeros((height, width), dtype=np.float32)

    idx = 0
    for row in range(num_tiles_y):
        for col in range(num_tiles_x):
            # Extract prediction tile
            pred_tile = predictions[idx]
            pred_tile = pred_tile.squeeze()  # Remove channel dimension if necessary
            h_start = row * tile_size[1]
            h_end = min((row + 1) * tile_size[1], height)
            w_start = col * tile_size[0]
            w_end = min((col + 1) * tile_size[0], width)

            pred_tile_cropped = pred_tile[:h_end - h_start, :w_end - w_start]

            full_prediction[h_start:h_end, w_start:w_end] = pred_tile_cropped
            idx += 1

    # Apply threshold
    full_prediction = (full_prediction > threshold).astype(np.uint8)

    # Update metadata
    meta.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.uint8
    })

    # Write to file
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(full_prediction, 1)

def main(input_directory, output_directory, threshold=0.5):
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all .tif files in the input directory
    for input_raster in glob.glob(os.path.join(input_directory, "*.tif")):
        # Get the base name of the input file without extension
        base_name = os.path.basename(input_raster)
        base_name_no_ext = os.path.splitext(base_name)[0]

        # Set the output path for the predicted mask
        output_raster = os.path.join(output_directory, f"{base_name_no_ext}_mask.tif")

        # Get metadata from the input image
        with rasterio.open(input_raster) as src:
            meta = src.meta.copy()
            width = src.width
            height = src.height
            tile_size = (512, 512)

        # Create TileGenerator
        tile_generator = TileGenerator(input_raster, tile_size=tile_size, batch_size=32)

        # Predict on tiles
        predictions = []
        for batch in tile_generator:
            pred = model.predict(batch)
            predictions.extend(pred)

        predictions = np.array(predictions)

        # Merge predictions and save the output image
        merge_predictions(predictions, width, height, tile_size, output_raster, meta, threshold)

if __name__ == "__main__":
    input_directory = r"C:\Users\admin\OneDrive\Bureau\Projet deep learning et documentations\Benchmarks\ORTHO\2022"
    output_directory = r"C:\Users\admin\OneDrive\Bureau\Projet deep learning et documentations\Benchmarks\Predictions\2022"
    main(input_directory, output_directory)