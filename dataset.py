"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import random

import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset

from utils import get_imagenames
from utils import open_image
from utils import open_sequence


# number of frames of each sequence to include in validation dataset
NUMFRXSEQ_VAL = 15
# pattern for name of validation sequence
VALSEQPATT = "*"


class TrainDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        gray_mode=False,
        sequence_length=5,
        crop_size=96,
    ):
        self._gray_mode = gray_mode
        self._crop_size = crop_size

        self._crop = torchvision.transforms.RandomCrop(size=self._crop_size)
        self._augments, self._augment_weights = self._create_augments()

        # Collect directories for noisy videos.
        noisy_dirs = sorted(glob.glob(
            os.path.join(dataset_root, "before", "*"),
        ))
        # Collect directories for the corresponding clean videos.
        clean_dirs = sorted(glob.glob(
            os.path.join(dataset_root, "after", "*"),
        ))

        imagenames_pairs = []
        keyframe_pad = sequence_length // 2
        for noisy_dir, clean_dir in zip(noisy_dirs, clean_dirs):
            # Collect noisy image filenames for a given video.
            noisy_imagenames = get_imagenames(noisy_dir)
            # Collect corresponding clean image filenames for a given video.
            clean_imagenames = get_imagenames(clean_dir)

            assert len(noisy_imagenames) == len(clean_imagenames), \
                "Number of images are not equal for noisy and clean pair"

            keyframe_start = keyframe_pad
            keyframe_end = len(noisy_imagenames) - keyframe_pad
            for keyframe_index in range(keyframe_start, keyframe_end):
                # For each timestamp of the video, we pick up a pair of noisy
                # images sequence and it corresponding clean image.
                noisy_start = keyframe_index - keyframe_pad
                noisy_end = keyframe_index + keyframe_pad + 1
                noisy_frames = noisy_imagenames[noisy_start:noisy_end]
                clean_frame = clean_imagenames[keyframe_index]
                imagenames_pairs.append((noisy_frames, clean_frame))

        self._imagenames_pairs = imagenames_pairs

    def __getitem__(self, index):
        imagenames_pair = self._imagenames_pairs[index]

        # Load noisy images sequence.
        images = [
            open_image(
                imagename,
                gray_mode=self._gray_mode,
                expand_if_needed=False,
                expand_axis0=False,
            )[0]
            for imagename in imagenames_pair[0]
        ]
        # Load corresponding clean image.
        images.append(open_image(
            imagenames_pair[1],
            gray_mode=self._gray_mode,
            expand_if_needed=False,
            expand_axis0=False,
        )[0])
        # Convert the images into Tensor.
        # Shape of the images Tensor: (sequence_length + 1, C, H, W)
        images = torch.from_numpy(np.array(images))

        # Apply random crop on both noisy images sequence and clean image.
        # Shape of the patches Tensor:
        # (sequence_length + 1, C, crop_size, crop_size)
        patches = self._crop(images)

        # Apply a random data augmentation on both noisy patches sequence and
        # clean patch.
        augment_func = random.choices(self._augments, self._augment_weights)[0]
        patches = augment_func(patches)

        # Get noisy patches.
        # Shape of the noisy patches Tensor:
        # (sequence_length, C, crop_size, crop_size)
        noisy_patches = patches[:-1]
        # Convert the shape of noisy pathces.
        # Shape of the noisy patches Tensor for now:
        # (sequence_length * C, crop_size, crop_size)
        noisy_patches = noisy_patches.view((-1, *noisy_patches.shape[2:]))
        # Get clean patch.
        # Shape of the noisy patch Tensor:
        # (C, crop_size, crop_size)
        clean_patch = patches[-1]

        return noisy_patches, clean_patch

    def __len__(self):
        return len(self._imagenames_pairs)

    def _create_augments(self):
        do_nothing = lambda x: x
        do_nothing.__name__ = "do_nothing"
        rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
        rot90.__name__ = "rot90"
        flipud = lambda x: torch.flip(x, dims=[2])
        flipud.__name__ = "flipup"
        rot90_flipud = \
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        rot90_flipud.__name__ = "rot90_flipud"
        rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
        rot180.__name__ = "rot180"
        rot180_flipud = \
            lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
        rot180_flipud.__name__ = "rot180_flipud"
        rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        rot270.__name__ = "rot270"
        rot270_flipud = \
            lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        rot270_flipud.__name__ = "rot270_flipud"
        add_csnt = lambda x: x + torch.normal(
            mean=torch.zeros(x.size()[0], 1, 1, 1),
            std=(5/255.),
        ).expand_as(x).to(x.device)
        add_csnt.__name__ = "add_csnt"

        # Define augmentations and their ferquency to be chosen.
        augments = [
            do_nothing,
            flipud,
            rot90,
            rot90_flipud,
            rot180,
            rot180_flipud,
            rot270,
            rot270_flipud,
            add_csnt,
        ]
        augment_weights = [
            32,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
        ]

        return augments, augment_weights


class ValDataset(Dataset):
    """Validation dataset. Loads all the images in the dataset folder on memory.
    """
    def __init__(
        self,
        valsetdir=None,
        gray_mode=False,
        num_input_frames=NUMFRXSEQ_VAL,
    ):
        self.gray_mode = gray_mode

        # Look for subdirs with individual sequences
        seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

        # open individual sequences and append them to the sequence list
        sequences = []
        for seq_dir in seqs_dirs:
            seq, _, _ = open_sequence(
                seq_dir,
                gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames,
            )
            # seq is [num_frames, C, H, W]
            sequences.append(seq)

        self.sequences = sequences

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)
