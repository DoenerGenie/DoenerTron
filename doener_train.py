import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define constants for dataset directories and model parameters
TRAIN_DIR = './data/train'
VALIDATION_DIR = './data/validation'
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
CLASS_NAMES = ['doener', 'duerum']

def load_dataset(directory):
    """Load an image dataset from a directory"""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CLASS_NAMES
    )

# Load train and validation datasets
train_dataset = load_dataset(TRAIN_DIR)
validation_dataset = load_dataset(VALIDATION_DIR)

print("Class names:", train_dataset.class_names)
drop_out = 0.6

def build_model():
    """Build a convolutional neural network model"""
    return tf.keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Rescaling(1./255),  # Normalization between 0 and 1
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Changed units from 64 to 128 for better accuracy
        layers.Dropout(drop_out),
        layers.Dense(len(CLASS_NAMES), activation='softmax')  # Changed activation function to softmax for multi-class classification
    ])

# Build and compile the model
model = build_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Update loss function to categorical cross-entropy
    metrics=['accuracy']
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=early_stopping_callback
)

# Save the trained model
model.save('./model/doener_model.keras')

# Plot training and validation accuracy curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accurary')
plt.legend()
plt.show()