from typing import List

class Track():
    def __init__(self, name, track):
        self.name = name
        self.track = track
        self.audio
        self.targets

class MyMusDB():
    def __init__(self, root_path, set_type) -> None:
        self.root_path = root_path
        self.tracks=[]
        self.__load_data__(set_type)

    def __load_data__(self, set_type):
        # here load files from root path to list of tracks
        if set_type == "train":
            pass
        elif set_type == "valid":
            pass
        elif set_type == "test":
            pass
        pass

