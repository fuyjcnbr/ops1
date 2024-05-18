

def test_model():
        from keras.utils import load_img
        from model.model import TrainModelDeep

        tm = TrainModelDeep("", "", "")
        tm.load_model("model/model.keras")

        img = load_img("tests/integrated/fig_8.png", color_mode="grayscale", target_size=(28, 28))
        code, name = tm.predict_from_picture(img)
        print(f"code={code}, name={name}")
        assert code == 8
