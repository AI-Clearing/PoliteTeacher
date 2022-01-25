CENTERMASK_HOME="/home/dfilipiak/projects/centermask2"
CONDA_HOME="/home/dfilipiak/anaconda3"

PARTITION="exp"
GPUS=2
OUTPUT_DIR_PREFIX="dev/"

OUTPUT_DIR="output/${OUTPUT_DIR_PREFIX}centermask-semisup/cps_centermask_R_50_FPN_ms_3x_sup1"
SUP_PERCENT="1.0"

source $CONDA_HOME/bin/activate centermask2

srun --partition=$PARTITION --gres=gpu:$GPUS \
  python $CENTERMASK_HOME/train_net.py \
    --num-gpus $GPUS \
    --config $CENTERMASK_HOME/configs/centermask-semisup/cps_centermask_R_50_FPN_ms_3x_sup1.yaml \
    OUTPUT_DIR $OUTPUT_DIR \
    DATALOADER.SUP_PERCENT $SUP_PERCENT

source $CONDA_HOME/bin/deactivate