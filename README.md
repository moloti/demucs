# Introduction

The repository is a fork made from facebook demucs repository.
The base for our implementation was the first version of demucs (branch v1).
The working branch is voices-separation.
## How to run

In order to run the solution locally follow the steps:

1. Install requirements from requirements.txt file
2. Download the LibriMix dataset.
3. Run the command: python -m demucs --musdb "path_to_librimix_dataset_wav8k_min" -b 16
4. Change the parameters according to all the possible ones that are in a demucs/parser.py file.

## demucs_clean

The new folder demucs_clean contains custom implementation of demucs model that uses the new pytorch_lightning package.
It is not a finished version.