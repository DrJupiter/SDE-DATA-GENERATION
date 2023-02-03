#!/bin/bash
#BSUB -J bsc_test
#BSUB -o bsc_test_%J.out
#BSUB -e bsc_test_%J.err
#BSUB -q hpc
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=5G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10
# end of BSUB options

echo "starts working"

source /zhome/59/e/156513/test-env/bin/activate 

python main.py

echo "finished working"