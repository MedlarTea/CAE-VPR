#!/bin/sh
ARCH=$1
DATADIR=/home/lab/data1/hanjing
if [ $# -ne 1 ]
  then
    echo "Arguments error: <ARCH>"
    exit 1
fi

python -u examples/cluster.py -d pitts -a ${ARCH} -b 64 --width 640 --height 480 --data-dir ${DATADIR}
