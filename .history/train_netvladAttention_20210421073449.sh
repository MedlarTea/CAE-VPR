# train the teacher net
# nohup bash scripts/train_attention_teacher.sh logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/checkpoint1.pth.tar >/dev/null 2>&1 &
# train the student net-- vprloss attloss teacher_resume Lvpr Latt 
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 10.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 20.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 30.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 40.0 >/dev/null 2>&1 &

bash scripts/train_attention_student.sh triplet mse logs/netVLAD/attention/student/robotcar30k-vgg16/conv5-conv6-vistypepow2-vprLtriplet-attLmse-lvpr1.0-latt1.0-lr0.001-neg10-tuple2/model_best.pth.tar 1.0 1.0
