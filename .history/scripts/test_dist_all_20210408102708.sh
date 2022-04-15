#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=1

# for visualizing
RETURNLAYER=$4  # choice=[conv6, conv5, ...]
VISTYPE=pow2  # choice=[abs_sum, pow2, max]


RESUME=$1
ARCH=vgg16
DATADIR=$2
DIMENSION=$3  # for autoencoder model evaluation




if [ $# -lt 1 ]
  then
    echo "Arguments error: <MODEL PATH>"
    echo "Optional arguments: <DATASET (default:pitts)> <SCALE (default:250k)>"
    exit 1
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/test_all.py --launcher pytorch \
    -a ${ARCH} \
    --test-batch-size 4 -j 8 \
    --vlad --reduction \
    --resume ${RESUME} \
    --data_dir ${DATADIR} \
    --dimension ${DIMENSION} \
    --return_layer ${RETURNLAYER} \
    --visType ${VISTYPE}\
    # --isvisualized 
    # --sync-gather
