
import io
import streamlit as st
from PIL import Image


TITLE_TEXT = "Классификация изображений одежды (fashion mnist)"
BUTTON_TEXT = "Распознать изображение"
UPLOADER_LABEL = "Выберите изображение для распознавания"
RESULTS_TEXT = "**Результаты распознавания:**"

IMAGE_W = 28
IMAGE_H = 28

if __name__ == "__main__":

    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
    import numpy as np


def load_image():
    uploaded_file = st.file_uploader(label=UPLOADER_LABEL)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# def preprocess_image(img):
#     img = img.resize((IMAGE_W, IMAGE_H))
#     x = image.img_to_array(img)
#     x = x.reshape(28, 28)
#     x = x / 255
#     # x = np.expand_dims(x, axis=0)
#     # x = preprocess_input(x)
#     return x


if __name__ == "__main__":

    @st.cache(allow_output_mutation=True)
    def load_model():
        from model.model import TrainModelDeep

        tm = TrainModelDeep("", "", "")
        tm.load_model("model/model.keras")
        # model = tm.probability_model
        return tm


    # def print_predictions(preds):
    #     print(f"preds={preds}")
    #     st.write(str(preds))
        # classes = decode_predictions(preds, top=3)[0]
        # for cl in classes:
        #     st.write(cl[1], cl[2])

    tm = load_model()
    st.title(TITLE_TEXT)
    img = load_image()
    result = st.button(BUTTON_TEXT)
    if result:
        code, name = tm.predict_from_picture(img)
        print(f"code={code}, name={name}")
        # x = preprocess_image(img)
        # preds = model.predict(x)
        # pred = tm.predict1D(img)
        st.write(RESULTS_TEXT)
        st.write(name)
        # print_predictions(preds)
