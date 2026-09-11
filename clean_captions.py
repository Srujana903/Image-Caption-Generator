import string

# -----------------------------
# Step 1: Load captions
# -----------------------------
def load_captions(filename):
    captions = {}

    with open(filename, "r", encoding="utf-8") as file:
        next(file)  # Skip header line

        for line in file:
            line = line.strip()

            if len(line) < 2:
                continue

            image, caption = line.split(",", 1)

            if image not in captions:
                captions[image] = []

            captions[image].append(caption)

    return captions


# -----------------------------
# Step 2: Clean captions
# -----------------------------
def clean_captions(captions):

    table = str.maketrans('', '', string.punctuation)

    for image in captions.keys():

        cleaned_list = []

        for caption in captions[image]:

            caption = caption.lower()

            words = caption.split()

            words = [word.translate(table) for word in words]

            words = [word for word in words if word.isalpha()]

            words = [word for word in words if len(word) > 1]

            caption = "startseq " + " ".join(words) + " endseq"

            cleaned_list.append(caption)

        captions[image] = cleaned_list


# -----------------------------
# Step 3: Run
# -----------------------------
captions = load_captions("captions.txt")

print("Before Cleaning:")
print(captions["1000268201_693b08cb0e.jpg"][0])

clean_captions(captions)

print("\nAfter Cleaning:")
print(captions["1000268201_693b08cb0e.jpg"][0])