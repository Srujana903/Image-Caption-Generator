import streamlit as st
import pickle
import numpy as np
import io

from PIL import Image, ImageDraw, ImageFont

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================================
# Load Tokenizer
# =====================================================
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# =====================================================
# Load Caption Model
# =====================================================
caption_model = load_model("caption_model.keras")

# =====================================================
# Load ResNet50
# =====================================================
base_model = ResNet50(weights="imagenet")

feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

MAX_LENGTH = 34

# =====================================================
# Extract Features
# =====================================================
def extract_features(image):

    image = image.convert("RGB")
    image = image.resize((224,224))

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = feature_model.predict(image, verbose=0)

    return feature

# =====================================================
# Word for ID
# =====================================================
def word_for_id(integer):

    for word, index in tokenizer.word_index.items():

        if index == integer:
            return word

    return None

# =====================================================
# Generate Caption
# =====================================================
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

    caption = in_text.replace("startseq","")
    caption = caption.replace("endseq","")

    return caption.strip()

# =====================================================
# Create Download Image
# =====================================================
def create_caption_image(image, caption):

    image = image.convert("RGB")

    width, height = image.size

    extra_height = 70

    final_img = Image.new(
        "RGB",
        (width, height + extra_height),
        "white"
    )

    final_img.paste(image, (0,0))

    draw = ImageDraw.Draw(final_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.rectangle(
        [(0,height),(width,height+extra_height)],
        fill="white"
    )

    draw.text(
        (20, height+20),
        "Caption: " + caption,
        fill="blue",
        font=font
    )

    return final_img

# =====================================================
# Streamlit Config
# =====================================================
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>

.stApp{
    background:white;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    color:#2563EB;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:25px;
}

.caption-box{
    background:#F4F8FF;
    border-left:6px solid #2563EB;
    border-radius:10px;
    padding:15px;
    color:#111827;
    font-size:18px;
    font-weight:600;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.markdown(
    '<div class="title">🖼️ Image Caption Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Generate captions for images using CNN + LSTM</div>',
    unsafe_allow_html=True
)

# =====================================================
# Sample Image
# =====================================================
st.subheader("📷 Sample Image")

sample = Image.open("images/1000268201_693b08cb0e.jpg")

left, center, right = st.columns([1,2,1])

with center:
    st.image(sample, caption="Sample Image", width=250)

st.markdown("""
<div class="caption-box">
📝 <b>Sample Caption</b><br><br>
A child in a pink dress is climbing up a set of stairs in an entry way.
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# Upload Section
# =====================================================
st.subheader("📂 Upload Your Own Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("✨ Generate Caption"):

        with st.spinner("Generating caption..."):

            feature = extract_features(image)

            caption = generate_caption(feature)

        st.success("Caption Generated Successfully!")

        st.markdown(
            f"""
            <div class="caption-box">
            📝 <b>Generated Caption</b><br><br>
            {caption}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create downloadable image
        download_img = create_caption_image(image, caption)

        buffer = io.BytesIO()

        download_img.save(buffer, format="PNG")

        st.download_button(
            label="⬇ Download Image with Caption",
            data=buffer.getvalue(),
            file_name="captioned_image.png",
            mime="image/png"
        )