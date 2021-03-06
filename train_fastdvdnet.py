"""
Trains a FastDVDnet model.

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset import TrainDataset
from dataset import ValDataset
from utils import svd_orthogonalization
from utils import close_logger
from utils import init_logging
from train_common import resume_training
from train_common import lr_scheduler
from train_common import log_train_psnr
from train_common import validate_and_log
from train_common import save_model_checkpoint


def main(**args):
    r"""Performs the main training loop
    """

    # Load dataset
    print('> Loading datasets ...')
    dataset_val = ValDataset(args['valset_dir'], gray_mode=False)
    dataset_train = TrainDataset(
        args['trainset_dir'],
        sequence_length=args['temp_patch_size'],
        step=3,
        crop_size=args['patch_size'],
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=args['num_workers'],
    )

    num_minibatches = int(len(dataset_train) // args['batch_size'])
    print("\t# of training samples: %d\n" % len(dataset_train))

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPU devices
    device_ids = [0]
    torch.backends.cudnn.benchmark = True  # CUDNN optimization

    # Create model
    model = FastDVDnet()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # Define loss
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # train
        for i, (img_train, gt_train) in enumerate(loader_train, 0):

            # Pre-training step
            model.train()

            # When optimizer = optim.Optimizer(net.parameters()) we only zero
            # the optim's grads
            optimizer.zero_grad()

            N, _, H, W = img_train.size()

            # Send tensors to GPU
            gt_train = gt_train.cuda(non_blocking=True)
            img_train = img_train.cuda(non_blocking=True)

            # Evaluate model and optimize it
            out_train = model(img_train)

            # Compute loss
            loss = criterion(gt_train, out_train) / (N * 2)
            loss.backward()
            optimizer.step()

            # Results
            if training_params['step'] % args['save_every'] == 0:
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Compute training PSNR
                log_train_psnr(
                    out_train,
                    gt_train,
                    loss,
                    writer,
                    epoch,
                    i,
                    num_minibatches,
                    training_params,
                )
            # update step counter
            training_params['step'] += 1

        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()

        # Validation and log images
        validate_and_log(
            model_temp=model,
            dataset_val=dataset_val,
            temp_psz=args['temp_patch_size'],
            writer=writer,
            epoch=epoch,
            lr=current_lr,
            logger=logger,
            trainimg=img_train,
        )

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
    ))

    # Close logger file
    close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the denoiser")

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        "--e",
        type=int,
        default=80,
        help="Number of total training epochs",
    )
    parser.add_argument(
        "--resume_training",
        "--r",
        action='store_true',
        help="resume training from a previous checkpoint",
    )
    parser.add_argument(
        "--milestone",
        nargs=2,
        type=int,
        default=[50, 60],
        help="When to decay learning rate; should be lower than 'epochs'",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--no_orthog",
        action='store_true',
        help="Don't perform orthogonalization as regularization",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Number of training steps to log psnr and "
             "perform orthogonalization",
    )
    parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=5,
        help="Number of training epochs to save state",
    )
    # Preprocessing parameters
    parser.add_argument(
        "--patch_size",
        "--p",
        type=int,
        default=96,
        help="Patch size",
    )
    parser.add_argument(
        "--temp_patch_size",
        "--tp",
        type=int,
        default=5,
        help="Temporal patch size",
    )
    parser.add_argument(
        "--num_workers",
        "--nw",
        type=int,
        default=2,
        help="Number of workers for training data loader",
    )
    # Dirs
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help='path of log files',
    )
    parser.add_argument(
        "--trainset_dir",
        type=str,
        default=None,
        help='path of trainset',
    )
    parser.add_argument(
        "--valset_dir",
        type=str,
        default=None,
        help='path of validation set',
    )
    argspar = parser.parse_args()

    print("\n### Training FastDVDnet denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
