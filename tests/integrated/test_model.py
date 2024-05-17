import numpy as np

from main import IMAGE_W, IMAGE_H, preprocess_image


def test_model():
        from keras.utils import load_img, img_to_array
        # from tensorflow.keras.applications.efficientnet import decode_predictions
        from model.model import TrainModelDeep


        tm = TrainModelDeep("", "", "")
        tm.load_model("model.keras")

        img = load_img("fig_0.png", color_mode="grayscale", target_size=(28, 28))
        code, name = tm.predict_from_picture(img)
        print(f"code={code}, name={name}")
        # # image = load_img("tests/integrated/fig_0.png", color_mode="rgb", target_size=(IMAGE_W, IMAGE_H))
        # # img = load_img("fig_0.png", color_mode="rgb", target_size=(28, 28))
        # img = load_img("fig_0.png", color_mode="grayscale", target_size=(28, 28))
        # # arr = img_to_array(image)
        # # x = np.expand_dims(arr, axis=0)
        # # preds = model.predict(x)
        # # classes = decode_predictions(preds, top=1)[0]
        #
        # x = preprocess_image(img)
        # preds = x.probability_model.predict(np.array([x]))
        # # preds = model.predict(x)
        # print(preds)
        assert code == 0
