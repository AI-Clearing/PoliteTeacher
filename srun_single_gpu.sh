source /home/azapala/anaconda3/bin/activate detectron2
cd ~/ssis/
srun  --partition=common  --qos=16gpu7d --time=3-0 --gres=gpu:"$1" --mem-per-gpu=12G python train_net.py --num-gpus="$1" "${@:2}"