"""
This scipr is to evaluate (calculate the evaluation criteria PLCC, SROCC, RMSE) on individual testing sets.
TRIQ should be first generated, and then the weights file is loaded.
"""
from models.triq_model import create_triq_model
from misc.imageset_handler import get_image_score_from_groups
from train.evaluation_spaq import ModelEvaluation
import time


def get_image_scores(mos_file):
    image_files = {}
    with open(mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split(',')
            image_file = content[0]
            score = float(content[1]) / 25. + 1
            image_files[image_file] = score

    return image_files


def val_main(args):
    if args['n_quality_levels'] > 1:
        using_single_mos = False
    else:
        using_single_mos = True

    if args['weights'] is not None:
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    val_folders = [r'F:\SPAG_image_quality_dataset\512\TestImage_512_new']
    spaq_mos_file = r'F:\SPAG_image_quality_dataset\512\image_mos.csv'
    image_scores = get_image_scores(spaq_mos_file)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    model = create_triq_model(n_quality_levels=5,
                              input_shape=(None, None, 3),
                              # transformer_params=[2, 32, 16, 64],
                              backbone=args['backbone'],
                              maximum_position_encoding=193)
    model.load_weights(args['weights'])

    evaluation = ModelEvaluation(model, test_image_files, test_scores, using_single_mos,
                                 imagenet_pretrain=imagenet_pretrain)
    plcc, srcc, rmse = evaluation.__evaluation__()


if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    args = {}
    args['n_quality_levels'] = 5
    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['fpn_type'] = 'fpn'
    args['weights'] = r'.\database\results_triq\triq_conv2D_all\triq_conv2D_all_distribution\38_0.8659_1.0442.h5'

    t_start = time.time()
    val_main(args)
    print('Elpased time: {}'.format(time.time() - t_start))