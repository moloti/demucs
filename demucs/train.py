# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import tqdm
import inspect
from demucs.stft_loss import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from .utils import apply_model, average_metric, center_trim


def train_model(epoch,
                dataset,
                model,
                criterion,
                optimizer,
                augment,
                repeat=1,
                device="cpu",
                seed=None,
                workers=8,
                world_size=1,
                batch_size=16):

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        sampler_epoch = epoch * repeat
        if seed is not None:
            sampler_epoch += seed * 1000
        sampler.set_epoch(sampler_epoch)
        batch_size //= world_size
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    current_loss = 0
    # for repetition in range(repeat):
    tq = tqdm.tqdm(loader,
                    ncols=120,
                    desc=f"[{epoch:03d}] train ({1}/{repeat})",
                    leave=True,
                    position=0,
                    file=sys.stdout,
                    unit=" batch")
    total_loss = 0
    for idx, (streams, _, _) in enumerate(tq):
        if len(streams) < batch_size:
            # skip uncomplete batch for augment.Remix to work properly
            continue
        streams = streams.to(device)
        sources = streams  # [:, 1:]
        # sources = augment(sources) # y 48000 -> 400000
        mix = sources.sum(dim=1) # x
        valid_length = model.valid_length(mix.shape[-1])
        delta = valid_length - mix.shape[-1]
        padded = F.pad(mix, (delta // 2, delta - delta // 2))
        estimates = model(padded) # pred_y 36524
        estimates = center_trim(estimates, sources)
        loss = criterion(estimates, sources)
        estimates = estimates[:, 1:]
        estimates = estimates.sum(dim=1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        current_loss = total_loss / (1 + idx)
        tq.set_postfix(loss=f"{current_loss:.4f}")

        # free some space before next round
        del streams, sources, mix, estimates, loss

    if world_size > 1:
        sampler.epoch += 1

    if world_size > 1:
        current_loss = average_metric(current_loss)
    return current_loss


def validate_model(epoch,
                   dataset,
                   model,
                   criterion,
                   device="cpu",
                   rank=0,
                   world_size=1,
                   shifts=0,
                   split=False):
    indexes = range(rank, len(dataset), world_size)
    tq = tqdm.tqdm(indexes,
                   ncols=120,
                   desc=f"[{epoch:03d}] valid",
                   leave=True,
                   position=0,
                   file=sys.stdout,
                   unit=" track")
    current_loss = 0
    for index in tq:
        streams,  _, _ = dataset[index]
        streams = streams.to(device)
        sources = streams  #
        mix = sources.sum(dim=0) # x
        
        estimates = apply_model(model, mix, shifts=shifts, split=split)
        # sources = center_trim(sources, estimates)
        loss = criterion(estimates, sources)
        current_loss += loss.item() / len(indexes)
        del estimates, streams, sources

    if world_size > 1:
        current_loss = average_metric(current_loss)
    return current_loss
