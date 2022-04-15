#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=1

# data
DATRDIR=/home/jing/Data/Dataset
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
LVPR=$4  # number
LATT=$5  # number
VPRLOSS=$1  # [triplet|sare_ind|sare_joint]
ATTLOSS=$2  # [mse|l1]

# training config
EPOCHS=30
TUPLESIZE=2
RESUME=$3  # logs/xxx/xxxx.tar.gz
MARGIN=0.1

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
-a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather --visType ${VISTYPE} --return_layer ${RETURNLAYER} --useSemantics \
--lr ${LR} --weight-decay 0.001 --vpr_loss_type ${VPRLOSS} --att_loss_type ${ATTLOSS} --lambda_vpr ${LVPR} --lambda_att ${LATT} \
--margin ${MARGIN} --eval-step 1 --epochs ${EPOCHS} --step-size 5  --resume ${RESUME} \
--logs-dir logs/netVLAD/attention/teacher/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${RETURNLAYER}-vistype${VISTYPE}-vprL${VPRLOSS}-attL${ATTLOSS}-lvpr${LVPR}-latt${LATT}-lr${LR}-neg${NEGNUM}-tuple[${TUPLESIZE}*${GPUS}] \
--data-dir ${DATRDIR}
     
