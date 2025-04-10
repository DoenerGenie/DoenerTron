import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_dir = './data/train'
validation_dir = './data/validation'


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150,150),
    batch_size=32,
    label_mode='categorical',
    class_names=['nicht_doener', 'doener', 'duerum']
)
print("Class names:", train_dataset.class_names)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(150,150),
    batch_size=32,
    label_mode='categorical',
    class_names=['nicht_doener', 'doener', 'duerum']
)

model = tf.keras.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Rescaling(1./255), # Normalisierung zwischen 0 und 1,
    layers.Conv2D(16,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Modell speichern (Architektur + Gewichte)
model.save('./model/doener_model.keras')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accurary')
plt.legend()
plt.show()