# train the teacher net
# nohup bash scripts/train_attention_teacher.sh logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/checkpoint1.pth.tar >/dev/null 2>&1 &
# train the student net-- vprloss attloss teacher_resume Lvpr Latt 
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 10.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 20.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 30.0 >/dev/null 2>&1 &
#nohup bash scripts/train_attention_student.sh triplet mse /home/lab/data1/hanjingModel/OpenIBL/logs/netVLAD/attention/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar 1.0 40.0 >/dev/null 2>&1 &

bash scripts/train_attention_student.sh triplet mse logs/netVLAD/attention/teacher/robotcar-vgg16/conv5-triplet-lr0.001-neg10-tuple2/model_best.pth.tar 1.0 5.0
