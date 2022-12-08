
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from demucs_clean.demucs_model import Demucs


def main(args):
    # Get parsed arguments
    dict_args = vars(args)

    # pick model
    if args.model == "demucs":
        model = Demucs(**dict_args)
    else:
        ## Add more models here
        raise ValueError("Model not supported")
    
    



if __name__ == "__main__":
    parser  = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # set model name to decide which model to use
    parser.add_argument("--model", type=str, default="demucs", help="Choose the model")

    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "demucs":
        parser = Demucs.add_model_specific_args(parser)
    else:
        ## Add more models here
        raise ValueError("Model not supported")
    
    args = parser.parse_args()

    main(args)






