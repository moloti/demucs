#BSUB -q gpua100
#BSUB -gpu "num=2"
#BSUB -J jobthomas
#BSUB -n 4
#BSUB -W 10:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -u s212816@dtu.dk
## -- send notification at start --
#BSUB -B
## -- send notification at completion--
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

## <loading of modules, dependencies etc.>
source venv/bin/activate
echo "Start training..."
cd demucs
python -m pip install -r requirements.txt
python -m demucs --musdb ../../../../../work3/projects/02456/project04/librimix/Libri2Mix/wav8k/min -b 256 --epochs 20
exit 0