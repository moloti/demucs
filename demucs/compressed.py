# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from concurrent import futures

from demucs.my_musdb import MyMusDB


from .audio import AudioFile


def get_musdb_tracks(root, subsets):
    # print(root, subsets)
    mus = MyMusDB(root, subsets)
    # print(mus.tracks)
    # TODO create a functionthat would return a dictionary of filenames and their paths
    return {track.name: track.path for track in mus.tracks}
    # mus = musdb.DB(root, *args, **kwargs)
    # return {track.name: track.path for track in mus}


class StemsSet:
    def __init__(self, tracks, folder_path, duration=None, stride=1, samplerate=44100, channels=1):

        self.metadata = []
        for name, path in tracks.items():
            # meta = dict(metadata[name])
            # we're missing duration on metadata of a track - might be needed to implement
            meta = {}
            meta["path"] = path
            meta["name"] = name
            meta["folder_path"] = folder_path
            self.metadata.append(meta)
            # if duration is not None and meta["duration"] < duration:
            #     raise ValueError(f"Track {name} duration is too small {meta['duration']}")
        self.metadata.sort(key=lambda x: x["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate

    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            return meta

    def __getitem__(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            streams, mean_streams,std_streams = AudioFile(meta["folder_path"], meta["name"]).read(seek_time=index * self.stride,
                                                   duration=self.duration,
                                                   channels=self.channels,
                                                   samplerate=self.samplerate)
            return (streams - mean_streams) / std_streams


def _get_track_metadata(path, filename):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path, filename)
    mix, _, _ = audio.read(streams=0, channels=1, samplerate=44100)
    return {"duration": audio.duration, "std": mix.std().item(), "mean": mix.mean().item()}


def build_metadata(tracks, workers=10):
    pendings = []
    with futures.ProcessPoolExecutor(workers) as pool:
        for name, path in tracks.items():
            pendings.append((name, pool.submit(_get_track_metadata, path, name)))
    return {name: p.result() for name, p in pendings}