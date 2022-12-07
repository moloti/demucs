
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from scipy.io import wavfile
import json
import logging
from pathlib import Path
import sys
sys.path.append('/Users/dg/Documents/Development/02456_Deep_Learning/demucs')

from demucs.my_musdb import MyMusDB

import tqdm
from demucs.compressed import StemsSet, StemsSetValidation, get_musdb_tracks, get_validation_tracks
from demucs.utils import apply_model, load_model
from torch.utils.data import DataLoader
import torch.hub
from pesq import pesq, NoUtterancesError
from pystoi import stoi

parser = argparse.ArgumentParser('demucs.evaluate', description='Evaluate Model Performance')

parser.add_argument('--musdb', help='The path to the directory where the clean and the noisy audio files are stored')
parser.add_argument('--dataset_type', default="dev", help='Choose between the different dataset sizes and types: dev, test, train-100, train-360')
parser.add_argument("--model_path", help="Path to local trained model.")
parser.add_argument("--evals", help="Path to where the testing results are stored.")
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument("--samplerate", type=int, default=8000)
parser.add_argument("--audio_channels", type=int, default=1)
parser.add_argument("--data_stride",
                        # default=44100,
                        default=8000,
                        type=int,
                        help="Stride for chunks, shorter = longer epochs")
parser.add_argument("--samples",
                        default=8000 * 6,
                        type=int,
                        help="number of samples to feed in")
parser.add_argument("-d",
                        "--device",
                        help="Device to train on, default is cuda if available else cpu", default='cpu')
parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')



logger = logging.getLogger(__name__)
rank = 0
world_size = 1
            

def evaluate(args, workers=2, model=None, data_loader=None, shifts=0, split=False, save=True):
    pesq = 0
    stoi = 0
    cnt = 0
    updates = 5

    if args.evals is None:
        eval_folder = args.musdb
    else:
        eval_folder = Path(args.evals)

    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    if(args.dataset_type):
        root_folder = args.dataset_type
    else:
        root_folder = "dev"

    duration = Fraction(args.samples + args.data_stride, args.samplerate)
    stride = Fraction(args.data_stride, args.samplerate)

    # Load model
    if model is None:
        model = load_model(args.model_path)

    model.eval()

    # Load data
    if data_loader is None and args.musdb:
        test_set_names = MyMusDB(args.musdb, "test", path=root_folder)
        test_set = StemsSet(get_musdb_tracks(args.musdb, subsets="test", root_folder=root_folder),
                            folder_path=args.musdb,
                            duration=duration,
                            stride=stride,
                            samplerate=args.samplerate,
                            channels=args.audio_channels)

        #loader = DataLoader(dataset, batch_size=1, num_workers=2)

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    pendings = []

    with ProcessPoolExecutor(workers or 1) as pool:
        for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout, position=0, leave=True):
                
            track, mean_track, std_track = test_set[index]
            musdb_track = test_set_names.tracks[index]

            mix = track.sum(dim=0)
            
            estimates = apply_model(model, mix.to(args.device), shifts=shifts, split=split)
            estimates = estimates * std_track + mean_track

            if save and '2902-9008-0000_6241-61943-0013.wav' in musdb_track.name:
                estimates_trans = estimates.transpose(1, 2)
                estimates_trans = estimates_trans.cpu().numpy()
                folder = eval_folder / "wav/test"
                folder.mkdir(exist_ok=True, parents=True)
                for estimate in estimates_trans:
                    wavfile.write(str(folder / (musdb_track.name)), args.data_stride, estimate)
            
            references = track
            references = references.numpy()
            estimates = estimates.cpu().numpy()


            if args.device == 'cpu':
                pendings.append((musdb_track.name, pool.submit(_estimate_and_run_metrics, references, estimates, args)))
            else:
                pendings.append((musdb_track.name, pool.submit(_run_metrics, references, estimates, args)))
            cnt += references.shape[0]
            del references, mix, estimates, track


        for pending in tqdm.tqdm(pendings, file=sys.stdout, position=0, leave=True):
            # Access the future and the name of the track
            track_name, future = pending
            try: 
                (pesq_i, stoi_i)  = future.result()
                pesq += pesq_i
                stoi += stoi_i
            except NoUtterancesError:
                logger.warning(f"Track {track_name} has no utterances, skipping")
                continue
    



    metrics = [pesq, stoi]
    pesq_final, stoi_final = average([m/cnt for m in metrics], cnt)
    logger.info(f'Test set performance:PESQ={pesq_final}, STOI={stoi_final}.')
    return pesq_final, stoi_final


def _estimate_and_run_metrics(references, estimates, args):
    return _run_metrics(references, estimates, args)


def _run_metrics(references, estimates, args):
    estimates = estimates[:, 0]
    references = references[:, 0]
    if args.pesq:
        pesq_i = get_pesq(references, estimates, sr=args.samplerate)
    else:
        pesq_i = 0
    stoi_i = get_stoi(references, estimates, sr=args.samplerate)
    return pesq_i, stoi_i
        

def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'nb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

def average(metrics, count=1.):
    """average.
    Average all the relevant metrices across processes
    `metrics`should be a 1D float32 vector. Returns the average of `metrics`
    over all hosts. You can use `count` to control the weight of each worker.
    """
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def main():
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()