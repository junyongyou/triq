from tensorflow.keras.callbacks import Callback
import numpy as np
from PIL import Image
import scipy.stats


class ModelEvaluationIQ(Callback):
    """
    Evaluation for IQA, the main function is to calculate PLCC, SROCC, RMSE and MAD after each train epoch.
    """
    def __init__(self, image_files, scores, using_single_mos, imagenet_pretrain=False):
        super(ModelEvaluationIQ, self).__init__()
        self.image_files = image_files
        self.scores = scores
        self.using_single_mos = using_single_mos
        self.imagenet_pretrain = imagenet_pretrain
        self.mos_scales = np.array([1, 2, 3, 4, 5])
        # self.live_images, self.live_scores = ImageProvider.generate_live_images()

    def __get_prediction_mos(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return prediction[0][0]

    def __get_prediction_distribution(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        prediction = np.sum(np.multiply(self.mos_scales, prediction[0]))
        return prediction

    def __evaluation__(self, image_files, scores, target_width=512, target_height=512, resize=False):
        predictions = []
        mos_scores = []

        for image_file, score in zip(image_files, scores):
            image = Image.open(image_file)
            if resize:
                image = np.asarray(image.resize((target_height, target_width)), dtype=np.float32)
            else:
                image = np.asarray(image, dtype=np.float32)
            if self.imagenet_pretrain:
                # ImageNnet normalization
                image /= 127.5
                image -= 1.
            else:
                # Normalization based on the combined database consisting of KonIQ-10k and LIVE-Wild datasets
                image[:, :, 0] -= 117.27205081970828
                image[:, :, 1] -= 106.23294835284031
                image[:, :, 2] -= 94.40750328714887
                image[:, :, 0] /= 59.112836751661085
                image[:, :, 1] /= 55.65498543815568
                image[:, :, 2] /= 54.9486100975773

            if self.using_single_mos:
                mos_scores.append(score)
                prediction = self.__get_prediction_mos(image)
            else:
                mos_scores.append(np.sum(np.multiply(self.mos_scales, score)))
                prediction = self.__get_prediction_distribution(image)

            predictions.append(prediction)

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SROCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SROCC, RMSE, MAD))
        return PLCC, SROCC, RMSE

    def on_epoch_end(self, epoch, logs=None):
        plcc, srcc, rmse = self.__evaluation__(self.image_files, self.scores)

        logs['plcc'] = plcc
        logs['srcc'] = srcc
        logs['rmse'] = rmse
