import numpy as np
from PIL import Image
import scipy.stats


class ModelEvaluation:
    """
    Evaluation the model, this script is actually a copy of evaluation callback.
    """
    def __init__(self, model, image_files, scores, using_single_mos, imagenet_pretrain=False):
        self.model = model
        self.image_files = image_files
        self.scores = scores
        self.using_single_mos = using_single_mos
        self.imagenet_pretrain = imagenet_pretrain
        self.mos_scales = np.array([1, 2, 3, 4, 5])

    def __get_prediction_mos(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return prediction[0][0]

    def __get_prediction_distribution(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        prediction = np.sum(np.multiply(self.mos_scales, prediction[0]))
        return prediction

    def __evaluation__(self, result_file=None):
        predictions = []
        mos_scores = []
        if result_file is not None:
            rf = open(result_file, 'w+')

        for image_file, score in zip(self.image_files, self.scores):
            image = Image.open(image_file)
            image = np.asarray(image, dtype=np.float32)
            if self.imagenet_pretrain: # image normalization using TF approach
                image /= 127.5
                image -= 1.
            else: # Image normalization by subtracting mean and dividing std
                image[:, :, 0] -= 117.27205081970828
                image[:, :, 1] -= 106.23294835284031
                image[:, :, 2] -= 94.40750328714887
                image[:, :, 0] /= 59.112836751661085
                image[:, :, 1] /= 55.65498543815568
                image[:, :, 2] /= 54.9486100975773

            if self.using_single_mos:
                prediction = self.__get_prediction_mos(image)
            else:
                score = np.sum(np.multiply(self.mos_scales, score))
                prediction = self.__get_prediction_distribution(image)

            mos_scores.append(score)

            predictions.append(prediction)
            print('Real score: {}, predicted: {}'.format(score, prediction))

            if result_file is not None:
                rf.write('{},{},{}\n'.format(image_file, score, prediction))

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))

        if result_file is not None:
            rf.close()
        return PLCC, SRCC, RMSE
