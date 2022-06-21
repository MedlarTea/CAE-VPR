#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=2

# for visualizing

VISTYPE=pow2  # choice=[abs_sum, pow2, max]

ARCH=$1 # vggConv or alexnetConv or netvlad or vgg16 of alexnet
RESUME=$5
DATADIR=$6  
FEATURES=4096  # for pca
DIMENSION=$4
D1=$2
D2=$3
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

CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/test_convauto.py --launcher pytorch \
    -a ${ARCH} \
    --test-batch-size 4 -j 8 \
    --resume ${RESUME} \
    --data_dir ${DATADIR} \
    --features ${FEATURES} --dimension ${DIMENSION} --d1 ${D1} --d2 ${D2}\
    --visType ${VISTYPE} --islayerNorm
    # --reduction --vlad --sync-gather
    # --vlad # in convAuto--vgg16 test, this must be cancled
    # --isvisualized 
    # --sync-gather
    # --return_layer ${RETURNLAYER}
