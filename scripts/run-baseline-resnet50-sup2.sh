CENTERMASK_HOME="/home/ptempczyk/df/centermask2"
CONDA_HOME="/home/ptempczyk/anaconda3"

source $CONDA_HOME/bin/activate centermask2

python $CENTERMASK_HOME/train_net.py \
  --num-gpus 8 \
  --config $CENTERMASK_HOME/configs/centermask-semisup/baseline_centermask_R_50_FPN_ms_3x_sup2.yaml

source $CONDA_HOME/bin/deactivate