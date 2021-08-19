#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import argparse
import os
import time

import cv2
import torch
import torch.nn as nn

from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr
from utils import init_logger_test
from utils import variable_to_cv2_image
from utils import remove_dataparallel_wrapper
from utils import open_sequence
from utils import close_logger


NUM_IN_FR_EXT = 5  # temporal size of patch
OUTIMGEXT = '.png'  # output images format


def save_out_seq(seqnoisy, seqclean, save_dir, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(
            save_dir,
            ('{}{}').format(idx, fext),
        )
        if len(suffix) == 0:
            out_name = os.path.join(
                save_dir,
                ('FastDVDnet_{}{}').format(idx, fext),
            )
        else:
            out_name = os.path.join(
                save_dir,
                ('FastDVDnet_{}_{}{}').format(suffix, idx, fext),
            )

        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)


def test_fastdvdnet(**args):
    """Denoises all sequences present in a given folder. Sequences must be
    stored as numbered image sequences.

    Inputs:
        args (dict) fields:
            "model_file": path to model
            "noisy_path": path to noisy sequence to denoise
            "clean_path": path to corresponding clean sequence
            "suffix": suffix to add to output name
            "max_num_fr_per_seq": max number of frames to load per sequence
            "dont_save_results: if True, don't save output images
            "no_gpu": if True, run model on CPU
            "save_path": where to save outputs as png
            "gray": if True, perform denoising of grayscale images instead of
                RGB
    """
    # Start time
    start_time = time.time()

    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    with torch.no_grad():
        # process data
        noisy_seq, _, _ = open_sequence(
            args['noisy_path'],
            args['gray'],
            expand_if_needed=False,
            max_num_fr=args['max_num_fr_per_seq'],
        )
        clean_seq, _, _ = open_sequence(
            args['clean_path'],
            args['gray'],
            expand_if_needed=False,
            max_num_fr=args['max_num_fr_per_seq'],
        )
        noisy_seq = torch.from_numpy(noisy_seq).to(device)
        # Clean sequence won't be fed into model, so keep it on same device.
        clean_seq = torch.from_numpy(clean_seq)
        seq_time = time.time()

        denframes = denoise_seq_fastdvdnet(
            noisy_seq,
            NUM_IN_FR_EXT,
            model_temp,
        )
        print(denframes.size())
        print(denframes.size())

    # Compute PSNR and log it
    stop_time = time.time()
    psnr = batch_psnr(denframes, clean_seq, 1.)
    psnr_noisy = batch_psnr(noisy_seq, clean_seq, 1.)
    loadtime = (seq_time - start_time)
    runtime = (stop_time - seq_time)
    seq_length = noisy_seq.size()[0]
    logger.info("Finished denoising")
    logger.info("Noisy path: {}".format(args['noisy_path']))
    logger.info("Clean path: {}".format(args['clean_path']))
    logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".
                format(seq_length, runtime, loadtime))
    logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(
        psnr_noisy,
        psnr,
    ))

    # Save outputs
    if not args['dont_save_results']:
        # Save sequence
        save_out_seq(
            noisy_seq,
            denframes,
            args['save_path'],
            args['suffix'],
            args['save_noisy'],
        )

    # close logger
    close_logger(logger)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Denoise a sequence with FastDVDnet",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="./model.pth",
        help='path to model of the pretrained denoiser',
    )
    parser.add_argument(
        "--noisy_path",
        type=str,
        default="./data/rgb/Kodak24",
        help='path to noisy sequence to denoise',
    )
    parser.add_argument(
        "--clean_path",
        type=str,
        default="./data/rgb/Kodak24",
        help='path to corresponding clean sequence',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help='suffix to add to output name',
    )
    parser.add_argument(
        "--max_num_fr_per_seq",
        type=int,
        default=25,
        help='max number of frames to load per sequence',
    )
    parser.add_argument(
        "--dont_save_results",
        action='store_true',
        help="don't save output images",
    )
    parser.add_argument(
        "--save_noisy",
        action='store_true',
        help="save noisy frames",
    )
    parser.add_argument(
        "--no_gpu",
        action='store_true',
        help="run model on CPU",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='./results',
        help='where to save outputs as png',
    )
    parser.add_argument(
        "--gray",
        action='store_true',
        help='perform denoising of grayscale images instead of RGB',
    )

    argspar = parser.parse_args()

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
