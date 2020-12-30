# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import scipy.stats
import numpy as np

import torch

from tqdm import tqdm
from torchvision import transforms

from vit_iqa.modeling_vit_iqa import VisionTransformer, CONFIGS
from torch.utils.data import DataLoader, SequentialSampler

from vit_iqa.group_generator_torch import GroupGenerator
from misc.imageset_handler import get_image_scores, get_image_score_from_groups


logger = logging.getLogger(__name__)
mos_scale = torch.from_numpy(np.array([1, 2, 3, 4, 5])).float()


def get_loader(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    args_dataset = {}
    args_dataset['batch_size'] = 1
    args_dataset['val_folders'] = [
        r'.\database\val\koniq_normal',
        # r'.\database\train\koniq_small',
        r'.\database\val\koniq_small',
        r'.\database\val\live'
    ]
    args_dataset['koniq_mos_file'] = r'.\database\koniq10k_images_scores.csv'
    args_dataset['live_mos_file'] = r'.\database\live_wild\live_mos.csv'
    image_scores = get_image_scores(args_dataset['koniq_mos_file'], args_dataset['live_mos_file'],
                                    using_single_mos=False)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(args_dataset['val_folders'], image_scores)
    testset = GroupGenerator(test_image_file_groups,
                             test_score_groups,
                             batch_size=args_dataset['batch_size'],
                             transform=transform_test)

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return test_loader

def eva_metrics(preds, labels):
    PLCC = scipy.stats.pearsonr(labels, preds)[0]
    SROCC = scipy.stats.spearmanr(labels, preds)[0]
    RMSE = np.sqrt(np.mean(np.subtract(preds, labels) ** 2))
    MAD = np.mean(np.abs(np.subtract(preds, labels)))
    # print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SROCC, RMSE, MAD))
    return PLCC, SROCC, RMSE, MAD


def setup(args):
    # Prepare model
    config = CONFIGS[args['model_type']]

    num_classes = 5
    # num_classes = 10 if args['dataset'] == "cifar10" else 100

    model = VisionTransformer(config, zero_head=True, num_classes=num_classes, load_transformer_weights=True)
    # model = torch.load(r'C:\vq_datasets\results\ViT_patch_mGPUs\test0_checkpoint_pretrain.bin')
    model.load_state_dict(torch.load(r'C:\vq_datasets\results\ViT_hybrid_mGPUs\test0_checkpoint.bin'))
    # model.load_state_dict(torch.load(r'C:\vq_datasets\results\ViT_patch_mGPUs\test0_checkpoint.bin'))
    # model.load_state_dict(torch.load(r'C:\vq_datasets\results\ViT_patch_mGPUs\test0_checkpoint_pretrain.bin'))
    model.eval()
    model.to(args['device'])
    return args, model


def valid(args, model):
    model.eval()
    all_preds, all_label = [], []
    test_loader = get_loader(args)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args['local_rank'] not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args['device']) for t in batch)
        x, y = batch
        x = torch.squeeze(x, dim=0)
        y = torch.squeeze(y, dim=0)
        with torch.no_grad():
            result = model(x)[0]
            preds = torch.matmul(result, mos_scale.to(args['device']))

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(torch.matmul(y, mos_scale.to(args['device'])).detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], torch.matmul(y, mos_scale.to(args['device'])).detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    plcc, srocc, rmse, mad = eva_metrics(all_preds, all_label)
    print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(plcc, srocc, rmse, mad))

    return plcc, srocc, rmse


def main():
    args = {}
    # args['model_type'] = 'ViT-B_16'
    args['model_type'] = 'R50-ViT-B_16'
    args['local_rank'] = -1

    # Setup CUDA, GPU & distributed training
    if args['local_rank'] == -1:
        # device = "cpu"
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # args['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device = torch.device("cuda")
        # torch.cuda.set_device(args['local_rank'])
        # device = torch.device("cuda", args['local_rank'])
        # torch.distributed.init_process_group(backend='nccl',
        #                                      timeout=timedelta(minutes=60))
        args['n_gpu'] = 2
    args['device'] = device

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    plcc, srocc, rmse = valid(args, model)


if __name__ == "__main__":
    main()
