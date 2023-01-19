from tensorflow.keras.callbacks import Callback
import numpy as np
from PIL import Image
import scipy.stats


class ModelEvaluationIQGenerator(Callback):
    """
    Evaluation for IQA, the main function is to calculate PLCC, SROCC, RMSE and MAD after each train epoch.
    """
    def __init__(self, val_generator, using_single_mos, evaluation_generator=None):
        super(ModelEvaluationIQGenerator, self).__init__()
        self.val_generator = val_generator
        self.evaluation_generator = evaluation_generator
        self.using_single_mos = using_single_mos
        self.mos_scales = np.array([1, 2, 3, 4, 5])

    def __get_prediction_mos(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return prediction[0][0]

    def __get_prediction_distribution(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        prediction = np.sum(np.multiply(self.mos_scales, prediction[0]))
        return prediction

    def __evaluation__(self, iq_generator):
        predictions = []
        mos_scores = []

        for j in range(iq_generator.__len__()):
            images, scores_batch = iq_generator.__getitem__(j)
            # mos_scores.extend(scores)

            prediction_batch = self.model.predict(images)
            prediction = []
            scores = []
            for i in range(prediction_batch.shape[0]):
                if self.using_single_mos:
                    prediction.append(prediction_batch[i,:][0])
                    scores.append(scores_batch[i])
                else:
                    prediction.append(np.sum(np.multiply(self.mos_scales, prediction_batch[i,:])))
                    scores.append(np.sum(np.multiply(self.mos_scales, scores_batch[i, :])))
#                 prediction.append(np.sum(np.multiply(self.mos_scales, prediction_batch[i,:])))
#                 scores.append(np.sum(np.multiply(self.mos_scales, scores_batch[i, :])))
            predictions.extend(prediction)
            mos_scores.extend(scores)

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SROCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SROCC, RMSE, MAD))
        return PLCC, SROCC, RMSE, MAD

    def on_epoch_end(self, epoch, logs=None):
        plcc, srcc, rmse, mad = self.__evaluation__(self.val_generator)

        logs['plcc'] = plcc
        logs['srcc'] = srcc
        logs['rmse'] = rmse

        if self.evaluation_generator:
            if epoch % 10 == 0:
                plcc_10th, srcc_10th, rmse_10th, mad_10th = self.__evaluation__(self.evaluation_generator)
                print('\nEpoch {}: PLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(epoch, plcc_10th, srcc_10th, rmse_10th, mad_10th))

