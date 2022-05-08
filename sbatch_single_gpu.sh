#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --job-name="$1"
#SBATCH --partition=common
#SBATCH --qos=8gpu7d
#SBATCH --gres=gpu:1
#SBATCH --time=4-0
#SBATCH --mem-per-cpu=5000
#SBATCH --export=ALL,PATH="/usr/local/cuda/bin:${PATH}"
#SBATCH --output=/home/azapala/outputs/"$1".txt

cd /home/azapala/ssis
source /home/azapala/anaconda3/bin/activate detectron2
python train_net.py "${@:2}"

exit 0
EOT
