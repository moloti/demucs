
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import logging
import sys
sys.path.append('/Users/dg/Documents/Development/02456_Deep_Learning/demucs')

import tqdm
from demucs.compressed import StemsSet, StemsSetValidation, get_musdb_tracks, get_validation_tracks
from demucs.utils import load_model
from torch.utils.data import DataLoader
import torch.hub
from pesq import pesq
from pystoi import stoi

parser = argparse.ArgumentParser('demucs.evaluate', description='Evaluate Model Performance')

parser.add_argument('--audio_dir', help='The path to the directory where the clean and the noisy audio files are stored')
parser.add_argument("-m", "--model_path", help="Path to local trained model.")
parser.add_argument("--eval_folder", help="Path to where the training results are stored.")
parser.add_argument("--samplerate", type=int, default=8000)
parser.add_argument("--audio_channels", type=int, default=1)
parser.add_argument("--data_stride",
                        # default=44100,
                        default=8000,
                        type=int,
                        help="Stride for chunks, shorter = longer epochs")
parser.add_argument("--samples",
                        default=8000 * 3,
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
            

def evaluate(args, workers=2, model=None, data_loader=None):
    pesq = 0
    stoi = 0
    cnt = 0
    updates = 5

    # output_dir = eval_folder / "results"
    # output_dir.mkdir(exist_ok=True, parents=True)
    # json_folder = eval_folder / "results/test"
    # json_folder.mkdir(exist_ok=True, parents=True)

    duration = Fraction(args.samples + args.data_stride, args.samplerate)
    stride = Fraction(args.data_stride, args.samplerate)

    # Load model
    if model is None:
        model = load_model(args.model_path)

    model.eval()

    # Load data
    if data_loader is None & args.audio_dir:
        dataset = StemsSetValidation(get_validation_tracks(args.audio_dir, subsets="clean"),
                            get_validation_tracks(args.audio_dir, subsets="noisy"),
                            folder_path=args.audio_dir,
                            #metadata,
                            duration=duration,
                            stride=stride,
                            samplerate=args.samplerate,
                            channels=args.audio_channels)

        loader = DataLoader(dataset, batch_size=1, num_workers=2)
    pendings = []


    with ProcessPoolExecutor(workers or 1) as pool:
        for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):

            for i, streams in enumerate(tq):
                
                noisy_streams = [x["noisy"]["streams"].to(args.device) for x in streams]
                clean_streams = [x["clean"]["streams"].to(args.device) for x in streams]


                if args.device == 'cpu':
                    pendings.append(pool.submit(_estimate_and_run_metrics, clean_streams, model, noisy_streams, args))
                else:
                    estimate = get_estimate(model, noisy_streams, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean_streams, estimate, args, model.sample_rate))
                cnt += clean_streams.shape[0]

            for pending in pendings:
                pesq_i, stoi_i = pending.result()
                pesq += pesq_i
                stoi += stoi_i

    metrics = [pesq, stoi]
    pesq_final, stoi_final = average([m/cnt for m in metrics], cnt)
    logger.info(f'Test set performance:PESQ={pesq_final}, STOI={stoi_final}.')
    return stoi_final, stoi_final



def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
    return estimate

def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate)


def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=sr)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=sr)
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
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
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