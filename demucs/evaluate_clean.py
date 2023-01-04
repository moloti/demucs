from pesq import pesq, NoUtterancesError
from pystoi import stoi
import fast_bss_eval
from torch.utils.data import DataLoader
import torch.hub



def run_metrics(references, estimates, samplerate, use_pesq=True):
    estimates = estimates[:, 0]
    references = references[:, 0]
    if use_pesq:
        pesq_i = get_pesq(references, estimates, sr=samplerate)
    else:
        pesq_i = 0
    stoi_i = get_stoi(references, estimates, sr=samplerate)
    sdr = get_sdr(references, estimates)
    si_sdr = get_sisdr(references, estimates)
    return pesq_i, stoi_i, sdr, si_sdr
        

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

def get_sdr(ref_sig, out_sig):
    """Calculate SDR.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SDR
    """
    sdr_val = 0
    for i in range(len(ref_sig)):
        sdr_val += fast_bss_eval.sdr(ref_sig[i], out_sig[i], False)[0]
    return sdr_val

def get_sisdr(ref_sig, out_sig):
    """Calculate SISDR.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SISDR
    """
    sisdr_val = 0
    for i in range(len(ref_sig)):
        sisdr_val += fast_bss_eval.si_sdr(ref_sig[i], out_sig[i])
    return sisdr_val

def average(metrics, count=1., world_size=1):
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