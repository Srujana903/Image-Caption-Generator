import string
import pickle
import numpy as np

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
# Dictionary → List
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
# Create sequences
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

                # Keep integer label (NOT one-hot)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

with open("features/features.pkl", "rb") as file:
    features = pickle.load(file)

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

captions = load_captions("captions.txt")
clean_captions(captions)

max_len = max_length(captions)

X1, X2, y = create_sequences(
    tokenizer,
    max_len,
    captions,
    features
)

print("Training data created successfully!\n")

print("X1 Shape :", X1.shape)
print("X2 Shape :", X2.shape)
print("y Shape  :", y.shape)

print("\nFirst y value:", y[0])