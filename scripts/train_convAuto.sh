#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=2

DATASET=robotcar
LAYERS=conv5
LR=0.001
BS=$5
WORKERS=8
HEIGHT=480
WIDTH=640
EPOCHS=50
EVALSTEP=5
RESUME=$7
DIMENSION=$4
DFEATURE=32
ISLAYERNORM=True
ARCH=$1
DATADIR=$6
D1=$2
D2=$3
while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/train_convauto.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} \
  --layers ${LAYERS} --syncbn \
  --width ${WIDTH} --height ${HEIGHT} -j ${WORKERS} --test-batch-size 16 \
  --lr ${LR} --weight-decay 0.001 --dimension ${DIMENSION} --d1 ${D1} --d2 ${D2}\
  --eval-step ${EVALSTEP} --epochs ${EPOCHS} --step-size 5 --bs ${BS}\
  --logs-dir logs/convAuto/${DATASET}/${ARCH}/lr${LR}-bs$[${BS}*${GPUS}]-islayernorm${ISLAYERNORM}-d1-$3-d2-$4-dimension$[${DFEATURE}*${DIMENSION}]\
  --data-dir ${DATADIR} \
  --vgg16_resume ${RESUME} --arch ${ARCH} \
  --islayerNorm
#   --sync-gather
