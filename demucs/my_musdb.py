from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

class Track():
    def __init__(self, name, path):
        self.name = name
        self.duration = 24000
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
        files = os.listdir(os.path.join(self.root_path, Path("dev/mix_single")))
        train_valid_files, test_files = train_test_split(files, test_size=0.1, random_state=42)
        train_files, valid_files = train_test_split(train_valid_files, test_size=0.2, random_state=42)

        if set_type == "train":
            for file_name in train_files:
                file_path = os.path.join(self.root_path, Path("dev/mix_single"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "valid":
            for file_name in valid_files:
                file_path = os.path.join(self.root_path, Path("dev/mix_single"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "test":
            for file_name in test_files:
                file_path = os.path.join(self.root_path, Path("dev/mix_single"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        

