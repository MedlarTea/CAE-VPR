#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=4

# data
DATRDIR=/home/lab/data1/hanjing
DATASET=pitts
SCALE=30k
HEIGHT=480
WIDTH=640
NEGNUM=10
WORKERS=8

# model
ARCH=vgg16
LAYERS=conv5
VISTYPE=abs_sum
RETURNLAYER=conv6  # 返回最后一层卷积核作为att

# optimizer
LR=0.001
VPRLOSS=triplet
RESUME=$1

# training config
EPOCHS=30
TUPLESIZE=1
MARGIN=0.1


# if [ $# -ne 1 ]
#   then
#     echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
#     exit 1
# fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_Attention.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} --width ${WIDTH} --height ${HEIGHT} --tuple-size ${TUPLESIZE} -j ${WORKERS} --neg-num ${NEGNUM} --test-batch-size 16 --cache-size 1000 \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather --visType ${VISTYPE} --return_layer ${RETURNLAYER} \
  --lr ${LR} --weight-decay 0.001 --vpr_loss_type ${VPRLOSS} \
  --margin ${MARGIN} --eval-step 1 --epochs ${EPOCHS} --step-size 5 --resume ${RESUME}\
  --logs-dir logs/netVLAD/attention/${DATASET}${SCALE}-${ARCH}/${LAYERS}-vprloss${VPRLOSS}-lr${LR}-neg${NEGNUM}-tuple$[${TUPLESIZE}*${GPUS}] \
  --data-dir ${DATRDIR}
