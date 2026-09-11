import pickle
import numpy as np

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------------------------
# Load Tokenizer
# -------------------------------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# -------------------------------------------------
# Load Trained Caption Model
# -------------------------------------------------
caption_model = load_model("caption_model.keras")

# -------------------------------------------------
# Load ResNet50 Feature Extractor
# -------------------------------------------------
base_model = ResNet50(weights="imagenet")

feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

MAX_LENGTH = 34


# -------------------------------------------------
# Extract Features from New Image
# -------------------------------------------------
def extract_features(filename):

    image = load_img(filename, target_size=(224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = feature_model.predict(image, verbose=0)

    return feature


# -------------------------------------------------
# Convert Integer ID → Word
# -------------------------------------------------
def word_for_id(integer):

    for word, index in tokenizer.word_index.items():

        if index == integer:
            return word

    return None


# -------------------------------------------------
# Generate Caption
# -------------------------------------------------
def generate_caption(photo):

    in_text = "startseq"

    for i in range(MAX_LENGTH):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=MAX_LENGTH,
            padding="post"
        )

        prediction = caption_model.predict(
            [photo, sequence],
            verbose=0
        )

        prediction = np.argmax(prediction)

        word = word_for_id(prediction)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq", "")
    caption = caption.replace("endseq", "")

    return caption.strip()


# -------------------------------------------------
# Test Image
# -------------------------------------------------

image_path = "Images/1000268201_693b08cb0e.jpg"

photo = extract_features(image_path)

caption = generate_caption(photo)

print("\nGenerated Caption:")
print(caption)