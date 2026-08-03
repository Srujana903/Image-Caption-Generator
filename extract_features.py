import os
import pickle
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet')

model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

print("ResNet50 Feature Extractor Loaded!")

image_folder = "Images"

features = {}

image_names = os.listdir(image_folder)

print("Total Images Found:", len(image_names))

for i, img_name in enumerate(image_names):

    img_path = os.path.join(image_folder, img_name)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)


        image_id = img_name.split(".")[0]

        features[image_id] = feature

        if (i + 1) % 500 == 0:
            print(f"{i+1} images completed")

    except Exception as e:
        print("Error:", img_name)

os.makedirs("features", exist_ok=True)

with open("features/features.pkl", "wb") as file:
    pickle.dump(features, file)

print("\nFeature Extraction Completed!")
print("Total Features Saved:", len(features))