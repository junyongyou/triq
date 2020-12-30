import logging

import torch
import torch.distributed

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from vit_iqa.group_generator_torch import GroupGenerator
from misc.imageset_handler import get_image_scores, get_image_score_from_groups

logger = logging.getLogger(__name__)


def get_loader(args):
    if args['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomResizedCrop((args['img_size'], args['img_size']), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        # transforms.Resize((args['img_size'], args['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Define train and validation data
    args_dataset = {}
    args_dataset['batch_size'] = args['batch_size']
    args_dataset['train_folders'] = [
        r'.\database\train\koniq_normal',
        r'.\database\train\koniq_small',
        r'.\database\train\live'
    ]
    args_dataset['val_folders'] = [
        r'.\database\val\koniq_normal',
        r'.\database\val\koniq_small',
        r'.\database\val\live'
    ]
    args_dataset['koniq_mos_file'] = r'.\database\koniq10k_images_scores.csv'
    args_dataset['live_mos_file'] = r'.\database\live_wild\live_mos.csv'
    image_scores = get_image_scores(args_dataset['koniq_mos_file'], args_dataset['live_mos_file'],
                                    using_single_mos=False)
    train_image_file_groups, train_score_groups = get_image_score_from_groups(args_dataset['train_folders'],
                                                                              image_scores)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(args_dataset['val_folders'], image_scores)
    trainset = GroupGenerator(train_image_file_groups,
                              train_score_groups,
                              batch_size=args_dataset['batch_size'],
                              transform=transform_train)
    testset = GroupGenerator(test_image_file_groups,
                             test_score_groups,
                             batch_size=args_dataset['batch_size'],
                             transform=transform_test)

    # if args['local_rank'] == 0:
    #     torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) #if args['local_rank'] == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=1,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
