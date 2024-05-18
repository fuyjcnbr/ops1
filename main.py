
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
    from keras.utils import load_img

    uploaded_file = st.file_uploader(label=UPLOADER_LABEL)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

        gray = Image.open(uploaded_file).convert("L")

        ar = np.asarray(gray)
        print(f"load_image ar.shape={ar.shape}")

        return gray #Image.open(io.BytesIO(image_data))
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

# def preprocess_image(img):
#     print(f"preprocess_image img.shape={img.shape}")
#     img = img.resize((IMAGE_W, IMAGE_H))
#     print(f"preprocess_image img.shape2={img.shape}")
#     x = image.img_to_array(img)
#     print(f"preprocess_image x.shape={x.shape}")
#     x = np.expand_dims(x, axis=0)
#     print(f"preprocess_image x2.shape={x.shape}")
#     x = preprocess_input(x)
#     print(f"preprocess_image x3.shape={x.shape}")
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
        # x = preprocess_image(img)
        code, name = tm.predict_from_picture(img)
        # code, name = tm.predict_from_picture(x)
        print(f"code={code}, name={name}")
        # x = preprocess_image(img)
        # preds = model.predict(x)
        # pred = tm.predict1D(img)
        st.write(RESULTS_TEXT)
        st.write(name)
        # print_predictions(preds)
