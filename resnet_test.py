from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os

base_model = ResNet50(weights='imagenet')

model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

print("ResNet50 Feature Extractor Loaded!")


img = image.load_img("images/sample.jpeg", target_size=(224,224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

features = model.predict(img_array)

print("Feature Shape:", features.shape)


os.makedirs("features", exist_ok=True)

np.save("features/sample_features.npy", features)

print("Features saved successfully!")