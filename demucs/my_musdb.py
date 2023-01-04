from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

class Track():
    def __init__(self, name, path, duration=48000):
        self.name = name
        self.duration = duration
        # self.track = track
        self.path = path
        self.audio = None
        self.targets = None

class MyMusDB():
    def __init__(self, root_path, set_type, path="dev") -> None:
        self.root_path = root_path
        self.tracks=[]
        self.__load_data__(set_type, path)

    def __len__(self):
        return len(self.tracks)

    def __load_data__(self, set_type, path):
        # here load files from root path to list of tracks
        dataset = "/mix_single"
        files = os.listdir(os.path.join(self.root_path, Path(path + dataset)))
        # I path is containing dev or tain-100 or train-360


        # if 'dev' in path or 'train-100' in path or 'train-360' in path:
        #     train_valid_files, test_files = train_test_split(files, test_size=0.1, random_state=42)
        #     train_files, valid_files = train_test_split(train_valid_files, test_size=0.2, random_state=42)
        # elif(path == 'test'):
        #     files = os.listdir(os.path.join(self.root_path, Path(path + dataset)))
        #     test_files = files


        if set_type == "train":
            # for file_name in train_files[:8]:
            for file_name in files:
                file_path = os.path.join(self.root_path, Path(path + dataset), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "valid":
            # for file_name in valid_files[:4]:
            for file_name in files:
                file_path = os.path.join(self.root_path, Path(path + dataset), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "test":
            # for file_name in test_files[:4]:
            for file_name in files:
                file_path = os.path.join(self.root_path, Path(path + dataset), Path(file_name))
                self.tracks.append(Track(file_name, file_path))

class ValidationData():
    def __init__(self, root_path, set_type) -> None:
        self.root_path = root_path
        self.tracks=[]
        self.__load_data__(set_type)

    def __len__(self):
        return len(self.tracks)

    def __load_data__(self, set_type):
        # here load files from root path to list of tracks

        if set_type == "noisy":
            for file_name in os.listdir(os.path.join(self.root_path, Path("dev/mix_single"))):
                file_path = os.path.join(self.root_path, Path("dev/mix_single"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))
        elif set_type == "clean":
            for file_name in os.listdir(os.path.join(self.root_path, Path("dev/s1"))):
                file_path = os.path.join(self.root_path, Path("dev/s1"), Path(file_name))
                self.tracks.append(Track(file_name, file_path))