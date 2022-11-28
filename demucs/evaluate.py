
import argparse
import logging
from demucs.utils import load_model
import torch.hub
from pesq import pesq
from pystoi import stoi

parser = argparse.ArgumentParser('demucs.evaluate', description='Evaluate Model Performance')

parser.add_argument('--audio_dir', help='The path to the directory where the clean and the noisy audio files are stored')
parser.add_argument("-m", "--model_path", help="Path to local trained model.")

logger = logging.getLogger(__name__)
            

def evaluate(args, model=None, data_loader=None):
    pesq = 0
    stoi = 0
    cnt = 0
    updates = 5

    # Load model
    if not model & args.model_path:
        model = load_model(args.model_path)

    model.eval()

    # Load data
    if data_loader is None:
        dataset
    
    
        


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


def main():
    args = parser.parse_args()

    #pesq, stoi = evaluate(args)
    #json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    #sys.stdout.write('\n')


if __name__ == '__main__':
    main()