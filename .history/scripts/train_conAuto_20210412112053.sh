#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

DATASET=robotcar
LAYERS=conv5
LR=0.001
BS=32
WORKERS=8
HEIGHT=480
WIDTH=640
EPOCHS=50
EVALSTEP=5
RESUME=$1
DIMENSION=$2
DFEATURE=32
ISLAYERNORM=True
ARCH=alexnet


while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/vggConvauto_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} \
  --layers ${LAYERS} --syncbn --sync-gather \
  --width ${WIDTH} --height ${HEIGHT} -j ${WORKERS} --test-batch-size 16 \
  --lr ${LR} --weight-decay 0.001 --dimension ${DIMENSION}\
  --eval-step ${EVALSTEP} --epochs ${EPOCHS} --step-size 5 --bs ${BS}\
  --logs-dir logs/${DATASET}-convAuto/${ARCH}/lr${LR}-bs$[${BS}*${GPUS}]-islayernorm${ISLAYERNORM}-dimension$[${DFEATURE}*${DIMENSION}]\
  --data-dir /home/lab/data1/hanjing\
  --islayerNorm ${ISLAYERNORM}\
  --vgg16_resume ${RESUME} --arch ${ARCH}
