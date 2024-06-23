
import streamlit as st
from PIL import Image


TITLE_TEXT = "Классификация изображений одежды (fashion mnist)"
BUTTON_TEXT = "Распознать изображение"
UPLOADER_LABEL = "Выберите изображение для распознавания"
RESULTS_TEXT = "**Результаты распознавания:**"

IMAGE_W = 28
IMAGE_H = 28

if __name__ == "__main__":

    import numpy as np


def load_image():

    uploaded_file = st.file_uploader(label=UPLOADER_LABEL)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

        gray = Image.open(uploaded_file).convert("L")

        ar = np.asarray(gray)
        print(f"load_image ar.shape={ar.shape}")

        return gray
    else:
        return None


if __name__ == "__main__":

    @st.cache(allow_output_mutation=True)
    def load_model():
        from model.model import TrainModelDeep

        tm = TrainModelDeep("", "", "")
        tm.load_model("model/model.keras")
        return tm


    tm = load_model()
    st.title(TITLE_TEXT)
    img = load_image()
    result = st.button(BUTTON_TEXT)
    if result:
        code, name = tm.predict_from_picture(img)
        print(f"code={code}, name={name}")
        st.write(RESULTS_TEXT)
        st.write(name)
