"""
This scipr is to evaluate (calculate the evaluation criteria PLCC, SROCC, RMSE) on individual testing sets.
TRIQ should be first generated, and then the weights file is loaded.
"""
from models.triq_model import create_triq_model
from misc.imageset_handler import get_image_scores, get_image_score_from_groups
from train.evaluation import ModelEvaluation
import time


def val_main(args):
    if args['n_quality_levels'] > 1:
        using_single_mos = False
    else:
        using_single_mos = True

    if args['weights'] is not None:
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    val_folders = [
                    r'.\database\val\koniq_normal',
                   r'.\database\val\koniq_small',
                   # r'.\database\train\live',
                   r'.\database\val\live']

    koniq_mos_file = r'.\database\koniq10k_images_scores.csv'
    live_mos_file = r'.\database\live_wild\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=using_single_mos)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    # validation_generator = GroupGenerator(test_image_file_groups,
    #                                       test_score_groups,
    #                                       batch_size=32,
    #                                       image_aug=False,
    #                                       imagenet_pretrain=False)

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
    args['backbone'] = 'resnet50'
    args['weights'] = r'.\TRIQ.h5'

    t_start = time.time()
    val_main(args)
    print('Elpased time: {}'.format(time.time() - t_start))