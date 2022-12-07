# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import sys
from concurrent import futures
from demucs.my_musdb import MyMusDB
from demucs.compressed import StemsSet, get_musdb_tracks


# import musdb
import museval
import torch as th
import tqdm
from scipy.io import wavfile
from torch import distributed

from .utils import apply_model


def evaluate(model,
             musdb_path,
             eval_folder,
             workers=2,
             device="cpu",
             rank=0,
             save=False,
             shifts=0,
             split=False,
             check=True,
             world_size=1):
    """
    Evaluate model using museval. Run the model
    on a single GPU, the bottleneck being the call to museval.
    """

    source_names = ["drums", "bass", "other", "vocals"]
    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    # test_set = MusDB(musdb_path, subsets=["test"])
    test_set_names = MyMusDB(musdb_path, "test")
    test_set = StemsSet(get_musdb_tracks(musdb_path, subsets="test"),
                            folder_path=musdb_path,
                            #metadata,
                            # duration=duration,
                            # stride=stride,
                            # samplerate=args.samplerate,
                            # channels=args.audio_channels
                            )



    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    pendings = []
    with futures.ProcessPoolExecutor(workers or 1) as pool:
        for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):
            track, mean_track, std_track = test_set[index]
            musdb_track = test_set_names.tracks[index]

            out = json_folder / f"{musdb_track.name}.json.gz"
            if out.exists():
                continue

            # mix = th.from_numpy(track).t().float()
            mix = track.sum(dim=0)
            # ref = mix.mean(dim=0)  # mono mixture

            estimates = apply_model(model, mix.to(device), shifts=shifts, split=split)
            estimates = estimates * std_track + mean_track

            estimates = estimates.transpose(1, 2)

            # estimates = estimates.transpose(1, 2)
            references = track
            references = references.numpy()
            estimates = estimates.cpu().numpy()
            if save:
                folder = eval_folder / "wav/test" / track.name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(source_names, estimates):
                    wavfile.write(str(folder / (name + ".wav")), 44100, estimate)

            if workers:
                pendings.append((track.name, pool.submit(museval.evaluate, references, estimates)))
            else:
                pendings.append((track.name, museval.evaluate(references, estimates)))
            del references, mix, estimates, track

        for track_name, pending in tqdm.tqdm(pendings, file=sys.stdout):
            if workers:
                pending = pending.result()
            sdr, isr, sir, sar = pending
            track_store = museval.TrackStore(win=8000*6, hop=8000*6, track_name=track_name)
            for idx, target in enumerate(source_names):
                values = {
                    "SDR": sdr[idx].tolist(),
                    "SIR": sir[idx].tolist(),
                    "ISR": isr[idx].tolist(),
                    "SAR": sar[idx].tolist()
                }

                track_store.add_target(target_name=target, values=values)
                json_path = json_folder / f"{track_name}.json.gz"
                gzip.open(json_path, "w").write(track_store.json.encode('utf-8'))
    if world_size > 1:
        distributed.barrier()
