import numpy as np
import glob
import cv2
import os
from PIL import Image
from models.triq_model import create_triq_model


# file_name = '120459577'

def load_model(args):
    model = create_triq_model(n_quality_levels=5,
                              input_shape=(None, None, 3),
                              backbone=args['backbone'],
                              maximum_position_encoding=193,
                              vis=True)
    model.load_weights(args['weights'])

    return model


def main(file_name):
# if __name__ == '__main__':

    args = {}
    args['n_quality_levels'] = 5
    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['fpn_type'] = 'fpn'
    args['weights'] = r'.\database\results_triq\triq_conv2d_koniq_small\triq_conv2D_koniq_small_distribution\18_0.8620_0.9222.h5'

    model = load_model(args)

    folder = r'.\database\train\koniq_small'
    files = glob.glob(os.path.join(folder, '*.jpg'))

    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]

        image_file = r'.\database\train\koniq_small\{}.jpg'.format(file_name)

        ori_image = Image.open(image_file)
        image = np.asarray(ori_image, dtype=np.float32)

        image /= 127.5
        image -= 1.

        prediction, att_mat = model.predict(np.expand_dims(image, axis=0))
        att_mat = np.stack(att_mat, axis=0).squeeze(axis=1)
        att_mat = np.mean(att_mat, axis=1)
        # att_mat = np.max(att_mat, axis=1)

        residual_att = np.eye(att_mat.shape[1])
        aug_att_mat = np.add(att_mat, residual_att)
        aug_att_mat = np.divide(aug_att_mat, np.expand_dims(np.sum(aug_att_mat, axis=-1), axis=-1))

        # Recursively multiply the weight matrices
        joint_attentions = np.zeros(aug_att_mat.shape)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.shape[0]):
            joint_attentions[n] = np.matmul(aug_att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        v_mask = v[0, 1:]
        mask = v_mask.reshape([12, 16])

        mask = cv2.resize(mask / mask.max(), ori_image.size)[..., np.newaxis]
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        # mask = cv2.resize(mask, ori_image.size)[..., np.newaxis]
        result = (mask * ori_image).astype("uint8")
        # result = np.matmul(mask, ori_image).astype("uint8")

        result_image = Image.fromarray(result)
        result_image.save(r'.\database\attention_masks_triq\{}_mask_triq.png'.format(file_name))
