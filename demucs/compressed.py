# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from concurrent import futures

from demucs.my_musdb import MyMusDB, ValidationData


from .audio import AudioFile


def get_musdb_tracks(root, subsets, root_folder="dev"):
    # print(root, subsets)
    mus = MyMusDB(root, subsets, root_folder)
    # print(mus.tracks)
    # TODO create a functionthat would return a dictionary of filenames and their paths
    return {track.name: (track.path, track.duration) for track in mus.tracks}
    # mus = musdb.DB(root, *args, **kwargs)
    # return {track.name: track.path for track in mus}



def get_validation_tracks(root, subsets):
    validation_set = ValidationData(root, subsets)
    tracks = {}
    for track in validation_set.tracks:
        tracks[track.name] = {"path": track.path, "duration": track.duration}
    return tracks


class StemsSet:
    def __init__(self, tracks, folder_path, duration=None, stride=1, samplerate=8000, channels=1):

        self.metadata = []
        for name, (path, duration) in tracks.items():
            # meta = dict(metadata[name])
            # we're missing duration on metadata of a track - might be needed to implement
            meta = {}
            meta["path"] = path
            meta["name"] = name
            meta["duration"] = duration
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
            return (streams - mean_streams) / std_streams, mean_streams, std_streams

class StemsSetValidation:
    def __init__(self, tracks_clean, tracks_noisy, folder_path, duration=None, stride=1, samplerate=8000, channels=1):

        self.metadata = []
        self.folder_path = folder_path
        self.datasets = {}
        self.datasets["clean"] = tracks_clean
        self.datasets["noisy"] = tracks_noisy
        tracks_clean_keys = list(tracks_clean)
        tracks_noisy_keys = list(tracks_noisy)

        for idx, (tracks_clean_key, tracks_noisy_key) in enumerate(zip(tracks_clean_keys, tracks_noisy_keys)):
            meta = self.create_metadata_set(tracks_clean_key, tracks_noisy_key)
            self.metadata.append(meta)

        self.metadata.sort(key=lambda x: x["clean"]["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate
    
    def create_metadate_item(self, track_key, dataset_type):
            meta = {}
            track = self.datasets[dataset_type][track_key]
            meta["path"] = track.path
            meta["name"] = track_key
            meta["duration"] = track.duration
            meta["folder_path"] = self.folder_path
            return meta

    
    def create_metadata_set(self, track_clean_key, track_noisy_key):
            meta_set = {}
            meta_set["clean"] = self.create_metadate_item(track_clean_key, "clean")
            meta_set["noisy"] = self.create_metadate_item(track_noisy_key, "noisy")
            return meta_set


    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["clean"]["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            return meta

    def convert_item(self, meta, index, dataset_type):
        stream_item = {}
        streams, mean_streams,std_streams = AudioFile(meta[dataset_type]["folder_path"], meta[dataset_type]["name"]).read(seek_time=index * self.stride,
                                                   duration=self.duration,
                                                   channels=self.channels,
                                                   samplerate=self.samplerate)
        stream_item["streams"] = (streams - mean_streams) / std_streams
        stream_item["mean_streams"] = mean_streams
        stream_item["std_streams"] = std_streams
        return stream_item


    def __getitem__(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            stream_item = {}
            stream_item["clean"] = self.convert_item(meta, index, "clean")
            stream_item["noisy"] = self.convert_item(meta, index, "noisy")
            return stream_item


def _get_track_metadata(path, filename):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path, filename)
    mix, _, _ = audio.read(streams=0, channels=1, samplerate=8000)
    return {"duration": audio.duration, "std": mix.std().item(), "mean": mix.mean().item()}


def build_metadata(tracks, workers=10):
    pendings = []
    with futures.ProcessPoolExecutor(workers) as pool:
        for name, path in tracks.items():
            pendings.append((name, pool.submit(_get_track_metadata, path, name)))
    return {name: p.result() for name, p in pendings}