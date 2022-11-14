from typing import List
from pathlib import Path
import os

class Track():
    def __init__(self, name, path):
        self.name = name
        # self.track = track
        self.path = path
        self.audio = None
        self.targets = None

class MyMusDB():
    def __init__(self, root_path, set_type) -> None:
        self.root_path = root_path
        self.tracks=[]
        self.__load_data__(set_type)

    def __load_data__(self, set_type):
        # here load files from root path to list of tracks
        if set_type == "train":
            for file_name in os.listdir(os.path.join(self.root_path, Path("dev/mix_single"))):
                file_path = os.path.join(self.root_path, Path("dev/mix_single"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "valid":
            for file_path in os.listdir(os.path.join(self.root_path, Path("train/mix_single"))):
                _, file_name = os.path.split(file_path)
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "test":
            pass
        pass

