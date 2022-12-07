# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

from dora.log import fatal
import torch as th
import tqdm
from scipy.io import wavfile

from .audio import AudioFile
from .utils import apply_model, load_model

BASE_URL = "https://dl.fbaipublicfiles.com/demucs/v2.0/"
PRETRAINED_MODELS = {
    'demucs.th': 'f6c4148ba0dc92242d82d7b3f2af55c77bd7cb4ff1a0a3028a523986f36a3cfd',
    'demucs.th.gz': 'e70767bfc9ce62c26c200477ea29a20290c708b210977e3ef2c75ace68ea4be1',
    'demucs_extra.th': '3331bcc5d09ba1d791c3cf851970242b0bb229ce81dbada557b6d39e8c6a6a87',
    'demucs_extra.th.gz': 'f9edcf7fe55ea5ac9161c813511991e4ba03188112fd26a9135bc9308902a094',
    'light.th': '79d1ee3c1541c729c552327756954340a1a46a11ce0009dea77dc583e4b6269c',
    'light.th.gz': '94c091021d8cdee0806b6df0afbeb59e73e989dbc2c16d2c1c370b2edce774fd',
    'light_extra.th': '9e9b4af564229c80cc73c95d02d2058235bb054c6874b3cba4d5b26943a5ddcb',
    'light_extra.th.gz': '48bb1a85f5ad0ca400512fcd0dcf91ec94e886a1602a552ee32133f5e09aeae0',
    'tasnet.th': 'be56693f6a5c4854b124f95bb9dd043f3167614898493738ab52e25648bec8a2',
    'tasnet_extra.th': '0ccbece3acd98785a367211c9c35b1eadae8d148b0d37fe5a5494d6d335269b5',
}


def download_file(url, target):
    """
    Download a file with a progress bar.

    Arguments:
        url (str): url to download
        target (Path): target path to write to
        sha256 (str or None): expected sha256 hexdigest of the file
    """
    def _download():
        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length', 0))

        with tqdm.tqdm(total=total_length, ncols=120, unit="B", unit_scale=True, position=0, leave=True) as bar:
            with open(target, "wb") as output:
                for data in response.iter_content(chunk_size=4096):
                    output.write(data)
                    bar.update(len(data))


def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'FFmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def main():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        help="Only separate audio into {STEM} and no_{STEM}. ")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--int24", action="store_true",
                       help="Save wav output as 24 bits wav.")
    group.add_argument("--float32", action="store_true",
                       help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    parser.add_argument("--mp3", action="store_true",
                        help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("-j", "--jobs",
                        default=0,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")

    args = parser.parse_args()

    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if args.segment is not None and args.segment < 8:
        fatal("Segment must greater than 8. ")

    if '..' in args.filename.replace("\\", "/").split("/"):
        fatal('".." must not appear in filename. ')

    if isinstance(model, BagOfModels):
        print(f"Selected model is a bag of {len(model.models)} models. "
              "You will see that many progress bars per track.")
        if args.segment is not None:
            for sub in model.models:
                sub.segment = args.segment
    else:
        if args.segment is not None:
            model.segment = args.segment

    model.cpu()
    model.eval()

    if args.stem is not None and args.stem not in model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                stem=args.stem, sources=', '.join(model.sources)))
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        print("here")
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        print(f"Separating track {track}")
        wav, _, _ = AudioFile(track, "somename").read(streams=0, samplerate=model.samplerate, channels=model.audio_channels).to(args.device)
        # Round to nearest short integer for compatibility with how MusDB load audio with stempeg.
        wav = (wav * 2**15).round() / 2**15
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav[None], device=args.device, shifts=args.shifts,
                              split=args.split, overlap=args.overlap, progress=True,
                              num_workers=args.jobs)[0]
        sources = sources * ref.std() + ref.mean()

        if args.mp3:
            ext = "mp3"
        else:
            ext = "wav"
        kwargs = {
            'samplerate': model.samplerate,
            'bitrate': args.mp3_bitrate,
            'clip': args.clip_mode,
            'as_float': args.float32,
            'bits_per_sample': 24 if args.int24 else 16,
        }
        if args.stem is None:
            for source, name in zip(sources, model.sources):
                stem = out / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                  trackext=track.name.rsplit(".", 1)[-1],
                                                  stem=name, ext=ext)
                stem.parent.mkdir(parents=True, exist_ok=True)
                save_audio(source, str(stem), **kwargs)
        else:
            sources = list(sources)
            stem = out / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                              trackext=track.name.rsplit(".", 1)[-1],
                                              stem=args.stem, ext=ext)
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(sources.pop(model.sources.index(args.stem)), str(stem), **kwargs)
            # Warning : after poping the stem, selected stem is no longer in the list 'sources'
            other_stem = th.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            stem = out / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                              trackext=track.name.rsplit(".", 1)[-1],
                                              stem="no_"+args.stem, ext=ext)
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(other_stem, str(stem), **kwargs)


if __name__ == "__main__":
    main()
