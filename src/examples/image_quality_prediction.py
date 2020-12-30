from models.triq_model import create_triq_model
import numpy as np
from PIL import Image


def predict_image_quality(model_weights_path, image_path):
    image = Image.open(image_path)
    image = np.asarray(image, dtype=np.float32)
    image /= 127.5
    image -= 1.

    model = create_triq_model(n_quality_levels=5)
    model.load_weights(model_weights_path)

    prediction = model.predict(np.expand_dims(image, axis=0))

    mos_scales = np.array([1, 2, 3, 4, 5])
    predicted_mos = (np.sum(np.multiply(mos_scales, prediction[0])))
    return predicted_mos


if __name__ == '__main__':
    # image_path = r'.\\sample_data\example_image_1 (mos=2.9).jpg'
    image_path = r'.\\sample_data\example_image_2 (mos=2.865).jpg'
    model_weights_path = r'C:\fish_lice_dataset\image_quality_koniq10k\TRIQ.h5'
    predict_mos = predict_image_quality(model_weights_path, image_path)
    print('Predicted MOS: {}'.format(predict_mos))