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
import collections
import os
import glob
import random

import torch
import torchvision
from torch.utils.data.dataset import Dataset

from utils import get_videonames
from utils import open_sequence


# number of frames of each sequence to include in validation dataset
NUMFRXSEQ_VAL = 15


SequencePair = collections.namedtuple("SequencePair", field_names=[
    "noisy_video",
    "noisy_frames",
    "clean_video",
    "clean_frame",
])


class TrainDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        gray_mode=False,
        sequence_length=5,
        step=3,
        crop_size=96,
        extensions=("mp4", "mov"),
    ):
        self._gray_mode = gray_mode
        self._step = step
        self._crop_size = crop_size
        self._extensions = extensions

        self._crop = torchvision.transforms.RandomCrop(size=self._crop_size)
        self._augments, self._augment_weights = self._create_augments()

        # Collect noisy videos.
        noisy_videos = get_videonames(
            os.path.join(dataset_root, "before"),
            self._extensions,
        )
        # Collect noisy videos.
        clean_videos = get_videonames(
            os.path.join(dataset_root, "after"),
            self._extensions,
        )

        sequence_pairs = []
        keyframe_pad = sequence_length // 2
        for noisy_video, clean_video in zip(noisy_videos, clean_videos):
            # Collct timestamps for noisy video.
            print("Load timestamps: {}".format(noisy_video))
            noisy_timestamps = torchvision.io.read_video_timestamps(
                noisy_video,
                pts_unit="sec",
            )[0]
            # Collct timestamps for clean video.
            print("Load timestamps: {}".format(clean_video))
            clean_timestamps = torchvision.io.read_video_timestamps(
                clean_video,
                pts_unit="sec",
            )[0]

            assert len(noisy_timestamps) == len(clean_timestamps), \
                "Number of frames are not equal for noisy and clean videos pair"

            keyframe_start = keyframe_pad
            keyframe_end = len(noisy_timestamps) - keyframe_pad
            for keyframe_index in range(keyframe_start, keyframe_end, self._step):
                # For each timestamp of the video, we pick up a pair of noisy
                # images sequence and it corresponding clean image.
                noisy_start = keyframe_index - keyframe_pad
                noisy_end = keyframe_index + keyframe_pad + 1
                noisy_frames = noisy_timestamps[noisy_start:noisy_end]
                clean_frame = clean_timestamps[keyframe_index]
                sequence_pairs.append(SequencePair(
                    noisy_video,
                    noisy_frames,
                    clean_video,
                    clean_frame,
                ))

        self._sequence_pairs = sequence_pairs

    def __getitem__(self, index):
        sequence_pair = self._sequence_pairs[index]

        # Load noisy images sequence.
        # Shape of the noisy images Tensor: (sequence_length, H, W, C)
        noisy_images = torchvision.io.read_video(
            sequence_pair.noisy_video,
            start_pts=sequence_pair.noisy_frames[0],
            end_pts=sequence_pair.noisy_frames[-1],
            pts_unit="sec",
        )[0]
        # Load clean image.
        # Shape of the clean image Tensor: (1, H, W, C)
        clean_image = torchvision.io.read_video(
            sequence_pair.clean_video,
            start_pts=sequence_pair.clean_frame,
            end_pts=sequence_pair.clean_frame,
            pts_unit="sec",
        )[0]
        # Concatenate noisy images and clean image to make following processing
        # flow easier.
        # Shape of the images Tensor: (sequence_length + 1, H, W, C)
        images = torch.cat([noisy_images, clean_image], axis=0)

        # Convert the images format from NHWC to HCHW.
        # Shape of current images Tensor: (sequence_length + 1, C, H, W)
        images = images.permute(0, 3, 1, 2)

        # Normalize the image values range from (0, 255) to (0.0, 1.0)
        images = images / 255.0

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
        noisy_patches = noisy_patches.contiguous().view(
            (-1, *noisy_patches.shape[2:]),
        )
        # Get clean patch.
        # Shape of the noisy patch Tensor:
        # (C, crop_size, crop_size)
        clean_patch = patches[-1]

        return noisy_patches, clean_patch

    def __len__(self):
        return len(self._sequence_pairs)

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
        dataset_root,
        gray_mode=False,
        num_input_frames=NUMFRXSEQ_VAL,
    ):
        self.gray_mode = gray_mode

        # Collect directories for noisy videos.
        noisy_dirs = sorted(glob.glob(
            os.path.join(dataset_root, "before", "*"),
        ))
        # Collect directories for the corresponding clean videos.
        clean_dirs = sorted(glob.glob(
            os.path.join(dataset_root, "after", "*"),
        ))

        # Open individual sequences for both noisy and clean data and append
        # them to the sequence list.
        sequences = []
        for noisy_dir, clean_dir in zip(noisy_dirs, clean_dirs):
            # Get a sequence of noisy images.
            # The shape is [num_frames, C, H, W]
            noisy_seq, _, _ = open_sequence(
                noisy_dir,
                gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames,
            )
            # Get a sequence of clean images.
            # The shape is [num_frames, C, H, W]
            clean_seq, _, _ = open_sequence(
                clean_dir,
                gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames,
            )

            sequences.append((noisy_seq, clean_seq))

        self._sequences = sequences

    def __getitem__(self, index):
        noisy_seq, clean_seq = self._sequences[index]

        return torch.from_numpy(noisy_seq), torch.from_numpy(clean_seq)

    def __len__(self):
        return len(self._sequences)
