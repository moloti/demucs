import os
from pathlib import Path
from typing import List, Tuple, Union
from collections import namedtuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
import torchaudio
from torch.nn import functional as F

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


class LibriMixDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, stride, duration,  **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.duration = duration

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LibriMix")
        parser.add_argument("--data_dir", type=int)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--duration", type=int, default=4)
        return parent_parser

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = ...
            self.val_dataset = ...

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

_TASKS_TO_MIXTURE = {
    "separate_clean": "mix_clean",
    "enhance_single": "mix_single",
    "enhance_both": "mix_both",
    "separate_noisy": "mix_both",
}

def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)

class LibriMixDataset(Dataset):
    """*LibriMix* :cite:`cosentino2020librimix` dataset.

    Functions inspired by torchaudio.datasets.LIBRIMIX.

    Args:
        root (str or Path): The path to the directory where the directory ``Libri2Mix`` or
            ``Libri3Mix`` is stored.
        subset (str, optional): The subset to use. Options: [``"train-360"``, ``"train-100"``,
            ``"dev"``, and ``"test"``] (Default: ``"train-360"``).
        num_speakers (int, optional): The number of speakers, which determines the directories
            to traverse. The Dataset will traverse ``s1`` to ``sN`` directories to collect
            N source audios. (Default: 2)
        sample_rate (int, optional): Sample rate of audio files. The ``sample_rate`` determines
            which subdirectory the audio are fetched. If any of the audio has a different sample
            rate, raises ``ValueError``. Options: [8000, 16000] (Default: 8000)
        task (str, optional): The task of LibriMix.
            Options: [``"enh_single"``, ``"enh_both"``, ``"sep_clean"``, ``"sep_noisy"``]
            (Default: ``"sep_clean"``)
        mode (str, optional): The mode when creating the mixture. If set to ``"min"``, the lengths of mixture
            and sources are the minimum length of all sources. If set to ``"max"``, the lengths of mixture and
            sources are zero padded to the maximum length of all sources.
            Options: [``"min"``, ``"max"``]
            (Default: ``"min"``)
    """
    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train-100",
        num_speakers: int = 1,
        sample_rate: int = 8000,
        task: str = "enhance_single",
        mode: str = "min",
        duration: int = 4,
        stride: int = 1,
    ):
        self.root = Path(root)
        self.num_cutted_examples = []
        self.duration = duration
        self.stride = stride

        if mode not in ["max", "min"]:
            raise ValueError(f'Expect ``mode`` to be one in ["min", "max"]. Found {mode}.')

        if sample_rate == 8000:
            mix_dir = self.root / "wav8k" / mode / subset
        elif sample_rate == 16000:
            mix_dir = self.root / "wav16k" / mode / subset
        else:
            raise ValueError(f"Expect ``sample_rate`` to be one in [8000, 16000]. Found {sample_rate}.")
        
        self.sample_rate = sample_rate
        self.task = task

        self.mix_dir = mix_dir / _TASKS_TO_MIXTURE[task]
        if task == "enhance_both":
            self.src_dirs = [(mix_dir / "mix_clean")]
        else:
            self.src_dirs = [(mix_dir / f"s{i+1}") for i in range(num_speakers)]

        self.files = [p.name for p in self.mix_dir.glob("*.wav")]
        self.files.sort()
        for file in self.files:
            file_length = get_info(os.path.relpath(self.mix_dir / file, self.root)).length
            if duration is None:
                examples = 1
            elif duration < file_length:
                examples = 0
            else:
                examples = (duration - file_length) // self.stride + 1
            self.num_cutted_examples.append(examples)



    def __len__(self):
        return sum(self.num_cutted_examples)


    def _load_audio(self, key: int, file, num_examples):
        """Load audio for the n-th sample from the dataset.

        Args:
            key (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            int:
                Sample rate
            Tensor:
                Mixed audio
            List of Tensor:
                List of source audios
        """
        metadata = self.get_metadata(key)
        mixed_path = metadata[1]
        srcs_paths = metadata[2]

        if key >= num_examples:
            key -= num_examples
            return
        num_frames = 0
        offset = 0
        if self.duration is not None:
            offset = key * self.stride
            num_frames = self.duration

        # TODO: Please refactor and simplify this part
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            mixed, _ = torchaudio.load(mixed_path, frame_offset=offset, num_frames=num_frames or -1)
            if num_frames:
                mixed = F.pad(mixed, (0, num_frames - mixed.shape[-1]))
            srcs = []
            for src in srcs_paths:
                src, _ = torchaudio.load(src, frame_offset=offset, num_frames=num_frames or -1)
                if num_frames:
                    src = F.pad(src, (0, num_frames - src.shape[-1]))
                
                srcs.append(src)
        else:
            mixed, _ = torchaudio.load(mixed_path, frame_offset=offset, num_frames=num_frames)
            if num_frames:
                mixed = F.pad(mixed, (0, num_frames - mixed.shape[-1]))
            srcs = []
            for src in srcs_paths:
                src, _ = torchaudio.load(src, frame_offset=offset, num_frames=num_frames)
                if num_frames:
                    src = F.pad(src, (0, num_frames - src.shape[-1]))
                srcs.append(src)

        
        return mixed, srcs


    def get_metadata(self, key: int):
        """Get metadata for the n-th sample from the dataset.

        Args:
            key (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            int:
                Sample rate
            str:
                Path to mixed audio
            List of str:
                List of paths to source audios
        """
        filename = self.files[key]
        mixed_path = os.path.relpath(self.mix_dir / filename, self.root)
        srcs_paths = []
        for dir_ in self.src_dirs:
            src = os.path.relpath(dir_ / filename, self.root)
            srcs_paths.append(src)

        return self.sample_rate, mixed_path, srcs_paths


    def __getitem__(self, key: int):
        for file, num_examples in zip(self.files, self.num_cutted_examples):
            return self._load_audio(key, file, num_examples)
        



    

    