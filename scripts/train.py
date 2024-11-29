import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from utils.gpu_setup import setup_gpus
from utils.metrics import jacard_coef
from utils.data_generator import DataGenerator
from models.unet import unet
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

# Configure GPU settings
setup_gpus(memory_limit=18000)

# Define metrics
metrics = ["Recall", jacard_coef]

# Paths
image_dir = r"D:\Deep_learning_final\Prediction_binaire_\img"
mask_dir = r"D:\Deep_learning_final\Prediction_binaire_\mask"
checkpoint_path = r"D:\Deep_learning_final\Checkpoints\model_checkpoint.h5"

# Load filenames
image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

# Split data
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Generators
train_generator = DataGenerator(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_filenames=train_files,
    batch_size=12,
    dim=(512, 512),
    n_channels=3,
    shuffle=True
)
val_generator = DataGenerator(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_filenames=val_files,
    batch_size=12,
    dim=(512, 512),
    n_channels=3,
    shuffle=False
)

# Model
if os.path.exists(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={'jacard_coef': jacard_coef})
    print("Checkpoint loaded.")
else:
    model = unet(metrics=metrics)
    print("No checkpoint found, starting from scratch.")

# Callbacks
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'
)
early_stopping_callback = EarlyStopping(patience=20, verbose=1)

# Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=70,
    verbose=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# Function to calculate performance metrics
def calculate_metrics(model, generator, threshold=0.5):
    accuracy_metric = BinaryAccuracy(threshold=threshold)
    precision_metric = Precision(thresholds=threshold)
    recall_metric = Recall(thresholds=threshold)

    for i in range(len(generator)):
        x_batch, y_true_batch = generator[i]
        y_pred_batch = model.predict(x_batch)

        accuracy_metric.update_state(y_true_batch, y_pred_batch)
        precision_metric.update_state(y_true_batch, y_pred_batch)
        recall_metric.update_state(y_true_batch, y_pred_batch)

    accuracy = accuracy_metric.result().numpy()
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1 Score calculation

    return accuracy, precision, recall, f1_score

# Evaluate model on validation set
val_accuracy, val_precision, val_recall, val_f1_score = calculate_metrics(model, val_generator)
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1 Score: {val_f1_score}")

# Confusion Matrix
def get_predictions_and_ground_truth(model, generator):
    predictions, ground_truth = [], []
    for i in range(len(generator)):
        x, y_true = generator[i]
        y_pred = model.predict(x)
        predictions.append(y_pred)
        ground_truth.append(y_true)

    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)

    # Flatten arrays for binary classification
    predictions = (predictions.flatten() > 0.5).astype(int)
    ground_truth = ground_truth.flatten().astype(int)

    return predictions, ground_truth

# Get predictions and ground truth
predictions, ground_truth = get_predictions_and_ground_truth(model, val_generator)

# Compute confusion matrix
cm = confusion_matrix(ground_truth, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualization of a prediction
def visualize_prediction(generator, model, idx):
    x_val, y_true = generator[idx]
    predicted_mask = model.predict(x_val)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(x_val[0])
    axs[0].title.set_text('Original Image')
    axs[0].axis('off')

    axs[1].imshow(y_true[0].squeeze(), cmap='gray')
    axs[1].title.set_text('Ground Truth Mask')
    axs[1].axis('off')

    predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)
    axs[2].imshow(predicted_mask_binary[0].squeeze(), cmap='gray')
    axs[2].title.set_text('Predicted Mask')
    axs[2].axis('off')
    
    plt.show()

batch_idx = random.randint(0, len(val_generator) - 1)
visualize_prediction(val_generator, model, batch_idx)