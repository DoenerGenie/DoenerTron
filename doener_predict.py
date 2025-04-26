import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Modell laden (für spätere Vorhersagen)
def load_saved_model(model_path):
    return tf.keras.models.load_model(model_path)


# Beispiel: Modell laden und Vorhersage machen
loaded_model = load_saved_model('./model/doener_model.keras')


def predict_image(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=[150, 150])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    print(prediction)
    if prediction[0, 0] > 0.8:
        return "Döner!"
    if prediction[0, 1] > 0.8:
        return "Dürum!"

    return "Weiss nicht!"


print("Erwartet: Weder noch. Ergebnis: ",
      predict_image('./data/validation/nicht_doener/IMG_20240411_161044.jpg', loaded_model))
print("Erwartet: Döner! Ergebnis: ", predict_image('./data/validation/doener/R-2.jpg', loaded_model))
print("Erwartet: Dürum! Ergebnis: ", predict_image('./data/validation/duerum/dueruem.jpg', loaded_model))
print("Erwartet: Weder noch! Ergebnis: ", predict_image('./data/validation/nicht_doener/IMG_20240511_091700.jpg', loaded_model))

