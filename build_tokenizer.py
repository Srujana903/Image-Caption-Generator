import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

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
        for cap in captions[key]:
            all_captions.append(cap)

    return all_captions


# -------------------------------------------------
# Maximum length
# -------------------------------------------------
def max_length(captions):

    lines = to_lines(captions)

    return max(len(line.split()) for line in lines)


# ============================
# MAIN PROGRAM
# ============================

captions = load_captions("captions.txt")

clean_captions(captions)

lines = to_lines(captions)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

vocab_size = len(tokenizer.word_index) + 1
max_len = max_length(captions)

print("Vocabulary Size :", vocab_size)
print("Maximum Length  :", max_len)

# ----------------------------------------
# Convert first caption into sequence
# ----------------------------------------

sample_caption = lines[0]

sequence = tokenizer.texts_to_sequences([sample_caption])[0]

print("\nOriginal Caption:")
print(sample_caption)

print("\nSequence:")
print(sequence)