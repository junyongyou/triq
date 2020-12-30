# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import scipy.stats
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from torch.nn.parallel.data_parallel import DataParallel as DDP
# from apex.parallel import DistributedDataParallel as DDP

from vit_iqa.modeling_vit_iqa import VisionTransformer, CONFIGS
# from ViT_pytorch.models.modeling import VisionTransformer, CONFIGS
from ViT_pytorch.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vit_iqa.iqa_data_utils import get_loader
from vit_iqa.modeling_vit_iqa import categorical_cross_entropy


logger = logging.getLogger(__name__)
mos_scale = torch.from_numpy(np.array([1, 2, 3, 4, 5])).float()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def eva_metrics(preds, labels):
    PLCC = scipy.stats.pearsonr(labels, preds)[0]
    SROCC = scipy.stats.spearmanr(labels, preds)[0]
    RMSE = np.sqrt(np.mean(np.subtract(preds, labels) ** 2))
    MAD = np.mean(np.abs(np.subtract(preds, labels)))
    print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SROCC, RMSE, MAD))
    return PLCC, SROCC, RMSE, MAD


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args['output_dir'], "%s_checkpoint.bin" % args['name'])
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args['output_dir'])


def setup(args):
    # Prepare model
    config = CONFIGS[args['model_type']]

    num_classes = 5
    # num_classes = 10 if args['dataset'] == "cifar10" else 100

    model = VisionTransformer(config, zero_head=True, num_classes=num_classes, load_transformer_weights=True)
    model.load_from(np.load(args['pretrained_dir']))
    if args['local_rank'] == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(args['device'])
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print('Number of parameters: {}'.format(num_params))
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(args['seed'])


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    model.eval()
    all_preds, all_label = [], []
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

            # eval_loss = loss_fct(logits, y)
            # result = F.softmax(output, dim=-1)
            # eval_loss = F.kl_div(result, y)
            eval_loss = categorical_cross_entropy(result, y)
            # eval_loss = F.nll_loss(y, result)
            # eval_loss = -(y * result).sum(-1).mean()
            eval_losses.update(eval_loss.item())

            preds = torch.matmul(result, mos_scale.to(args['device']))
            # preds = torch.argmax(result, dim=-1)

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
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    plcc, srocc, rmse, mad = eva_metrics(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid PLCC: %2.5f" % plcc)

    writer.add_scalar("test/accuracy", scalar_value=plcc, global_step=global_step)
    return plcc, srocc, rmse


def train(args, model):
    """ Train the model """
    if args['local_rank'] in [-1, 0]:
        os.makedirs(args['output_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args['name']))

    args['train_batch_size'] = args['train_batch_size'] // args['gradient_accumulation_steps']

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    train_set = train_loader.dataset

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args['learning_rate'],
                                momentum=0.9,
                                weight_decay=args['weight_decay'])
    t_total = args['num_steps']
    if args['decay_type'] == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)

    # if args['fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args['fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # # Distributed training
    # if args['local_rank'] != -1:
    #     model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args['num_steps'])
    logger.info("  Instantaneous batch size per GPU = %d", args['train_batch_size'])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args['train_batch_size'] * args['gradient_accumulation_steps'] * (
                    torch.distributed.get_world_size() if args['local_rank'] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_plcc = 0, 0
    epoch = 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args['local_rank'] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args['device']) for t in batch)
            x, y = batch
            x = torch.squeeze(x, dim=0)
            y = torch.squeeze(y, dim=0)
            loss = model(x, y)

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            # if args['fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                losses.update(loss.item()*args['gradient_accumulation_steps'])
                # if args['fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args['local_rank'] in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args['eval_every'] == 0 and args['local_rank'] in [-1, 0]:
                    plcc, srocc, rmse = valid(args, model, writer, test_loader, global_step)

                    if best_plcc < plcc:
                        save_model(args, model)
                        best_plcc = plcc
                        print('Improved, Epoch: {}, PLCC: {}, SROCC: {}, RMSE: {}'.format(epoch, plcc, srocc, rmse))
                    else:
                        print('Not improved, Epoch: {}, best PLCC: {}'.format(epoch, best_plcc))
                    model.train()
                    epoch += 1

                if global_step % t_total == 0:
                    break

        train_set.shuffle_dataset()
        losses.reset()
        if global_step % t_total == 0:
            break

    if args['local_rank'] in [-1, 0]:
        writer.close()
    logger.info("Best PLCC: \t%f" % best_plcc)
    logger.info("End Training!")


def main():
    # parser = argparse.ArgumentParser()
    # # Required parameters
    # parser.add_argument("--name", required=True,
    #                     help="Name of this run. Used for monitoring.")
    # parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
    #                     help="Which downstream task.")
    # parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
    #                                              "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
    #                     default="ViT-B_16",
    #                     help="Which variant to use.")
    # parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
    #                     help="Where to search for pretrained ViT models.")
    # parser.add_argument("--output_dir", default="output", type=str,
    #                     help="The output directory where checkpoints will be written.")
    #
    # parser.add_argument("--img_size", default=224, type=int,
    #                     help="Resolution size")
    # parser.add_argument("--train_batch_size", default=512, type=int,
    #                     help="Total batch size for training.")
    # parser.add_argument("--eval_batch_size", default=64, type=int,
    #                     help="Total batch size for eval.")
    # parser.add_argument("--eval_every", default=100, type=int,
    #                     help="Run prediction on validation set every so many steps."
    #                          "Will always run one evaluation at the end of training.")
    #
    # parser.add_argument("--learning_rate", default=3e-2, type=float,
    #                     help="The initial learning rate for SGD.")
    # parser.add_argument("--weight_decay", default=0, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--num_steps", default=10000, type=int,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
    #                     help="How to decay the learning rate.")
    # parser.add_argument("--warmup_steps", default=500, type=int,
    #                     help="Step of training to perform learning rate warmup for.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    #
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="local_rank for distributed training on gpus")
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # # parser.add_argument('--fp16_opt_level', type=str, default='O2',
    # #                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    # #                          "See details at https://nvidia.github.io/apex/amp.html")
    # parser.add_argument('--loss_scale', type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    # args = parser.parse_args()
    
    args = {}
    args['name'] = 'test0'
    args['dataset'] = 'cifar10'
    args['model_type'] = 'R50-ViT-B_16'
    # args['model_type'] = 'R50-ViT-Simple'
    # args['model_type'] = 'ViT-B_16'
    # args['pretrained_dir'] = r'.\pretrained_weights\imagenet21k+imagenet2012_ViT-B_16.npz'
    args['pretrained_dir'] = r'.\pretrained_weights\imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    args['output_dir'] = r'C:\vq_datasets\results\ViT'
    args['train_batch_size'] = 16
    args['eval_batch_size'] = 16
    args['eval_every'] = 1079 #4462
    args['learning_rate'] = 1e-4/2
    args['weight_decay'] = 0
    args['num_steps'] = 1079 * 200#4462 * 100
    args['decay_type'] = 'cosine'
    args['warmup_steps'] = 1079 * 20 # 4462 * 10
    args['max_grad_norm'] = 1.
    args['local_rank'] = -1
    args['seed'] = 32
    args['gradient_accumulation_steps'] = 1
    args['fp16'] = False
    args['loss_scale'] = 0

    # Setup CUDA, GPU & distributed training
    if args['local_rank'] == -1:
        # device = "cpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args['n_gpu'] = torch.cuda.device_count()
    elif args['local_rank'] == 0:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args['n_gpu'] = 1
    else:
        pass
    args['device'] = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args['local_rank'] in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args['local_rank'], args['device'], args['n_gpu'], bool(args['local_rank'] != -1), args['fp16']))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
