import string
import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import Embedding, Dropout, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Load captions
# -------------------------------------------------
def load_captions(filename):
    captions = {}

    with open(filename, "r", encoding="utf-8") as file:
        next(file)

        for line in file:
            line = line.strip()

            if len(line) < 2:
                continue

            image, caption = line.split(",", 1)
            image = image.split(".")[0]

            if image not in captions:
                captions[image] = []

            captions[image].append(caption)

    return captions


# -------------------------------------------------
# Clean captions
# -------------------------------------------------
def clean_captions(captions):

    table = str.maketrans('', '', string.punctuation)

    for image in captions.keys():

        cleaned = []

        for caption in captions[image]:

            caption = caption.lower()

            words = caption.split()

            words = [w.translate(table) for w in words]

            words = [w for w in words if w.isalpha()]

            words = [w for w in words if len(w) > 1]

            caption = "startseq " + " ".join(words) + " endseq"

            cleaned.append(caption)

        captions[image] = cleaned


# -------------------------------------------------
# Convert dictionary to list
# -------------------------------------------------
def to_lines(captions):

    all_captions = []

    for key in captions.keys():
        all_captions.extend(captions[key])

    return all_captions


# -------------------------------------------------
# Maximum length
# -------------------------------------------------
def max_length(captions):

    lines = to_lines(captions)

    return max(len(line.split()) for line in lines)


# -------------------------------------------------
# Create training data
# -------------------------------------------------
def create_sequences(tokenizer, max_len, captions, features):

    X1, X2, y = [], [], []

    for image_id, caps in captions.items():

        if image_id not in features:
            continue

        feature = features[image_id][0]

        for caption in caps:

            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):

                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences(
                    [in_seq],
                    maxlen=max_len,
                    padding="post"
                )[0]

                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


# =================================================
# Load data
# =================================================

with open("features/features.pkl", "rb") as file:
    features = pickle.load(file)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

captions = load_captions("captions.txt")
clean_captions(captions)

VOCAB_SIZE = len(tokenizer.word_index) + 1
MAX_LENGTH = max_length(captions)

X1, X2, y = create_sequences(
    tokenizer,
    MAX_LENGTH,
    captions,
    features
)

print("Training samples:", len(X1))


# =================================================
# Build Model
# =================================================

# Image branch
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

# Text branch
inputs2 = Input(shape=(MAX_LENGTH,))
se1 = Embedding(VOCAB_SIZE, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Merge
decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)

outputs = Dense(VOCAB_SIZE, activation="softmax")(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# =================================================
# Train
# =================================================

model.fit(
    [X1, X2],
    y,
    epochs=10,
    batch_size=64,
    verbose=1
)

# =================================================
# Save model
# =================================================

model.save("caption_model.keras")

print("\nModel saved successfully!")