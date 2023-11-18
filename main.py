from transformers import pipeline
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
import streamlit as st
import soundfile as sf
import io

st.title("From image to audio")
st.header("This neural network voices what is depicted in your image")
st.subheader("To try it, upload your image by clicking on the button below")



@st.cache_resource
def load_model():
    return BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

@st.cache_resource
def load_processor():
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
processor = load_processor()
model = load_model()
TEXT = "a potography of"

uploadImage = st.file_uploader("Choose image")



if uploadImage is not None:
    raw_image = Image.open(uploadImage).convert("RGB")

    inputs = processor(raw_image, TEXT, return_tensors="pt")

    out = model.generate(**inputs)
    st.text(processor.decode(out[0], skip_special_tokens=True))

    ImageToTextInput = processor.decode(out[0], skip_special_tokens=True)
else:
    st.error("Upload image!")
