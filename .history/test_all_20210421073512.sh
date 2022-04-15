# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/conv5-triplet-lr0.001-neg5-tuple4/model_best.pth.tar ~/Data/Dataset/ 512 conv5
# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/conv5-triplet-lr0.001-neg5-tuple4/model_best.pth.tar ~/Data/Dataset/ 512 conv5
# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/author/conv5-sare_ind-lr0.001-tuple4-SFRS/model_best.pth.tar ~/Data/Dataset/
# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/author/conv5-sare_ind-lr0.001-tuple4/model_best.pth.tar ~/Data/Dataset/


# # for alexnetConvAuto
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension16384/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension8192/model_best.pth.tar /home/lab/data1/hanjing 256 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension4096/model_best.pth.tar /home/lab/data1/hanjing 128 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension2048/model_best.pth.tar /home/lab/data1/hanjing 64 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension1024/model_best.pth.tar /home/lab/data1/hanjing 32 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension512/model_best.pth.tar /home/lab/data1/hanjing 16 conv5 alexnet
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/alexnet/lr0.001-bs128-islayernormTrue-dimension256/model_best.pth.tar /home/lab/data1/hanjing 8 conv5 alexnet
# # for vggConvAuto
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension16384/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension8192/model_best.pth.tar /home/lab/data1/hanjing 256 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension4096/model_best.pth.tar /home/lab/data1/hanjing 128 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension2048/model_best.pth.tar /home/lab/data1/hanjing 64 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension1024/model_best.pth.tar /home/lab/data1/hanjing 32 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension512/model_best.pth.tar /home/lab/data1/hanjing 16 conv5 vgg16
# bash scripts/test_dist_all.sh logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension256/model_best.pth.tar /home/lab/data1/hanjing 8 conv5 vgg16
# for vgg16
# bash scripts/test_dist_all.sh logs/vgg16/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16
# for alexnet, without loading checkpoints
# bash scripts/test_dist_all.sh logs/alexnet/imagenet_matconvnet_alex.pth /home/lab/data1/hanjing 512 conv5 alexnet
# for netvlad
# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/conv5-triplet-lr0.001-neg1-tuple4/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16


# bash scripts/test_dist_all.sh logs/netVLAD/pitts30k-vgg16/author/conv5-sare_ind-lr0.001-tuple4-SFRS/model_best.pth.tar ~/Data/Dataset/ 512 conv5
# bash scripts/test_dist_all.sh logs/alexnet/imagenet_matconvnet_alex.pth /home/lab/data1/hanjing 512 conv5

# for robotcar-trained netvlad
# bash scripts/test_dist_all.sh logs/netVLAD/attention/student/robotcar30k-vgg16/withoutConv/conv5-conv6-vistypepow2-vprLtriplet-attLmse-lvpr1.0-latt1.0-lr0.001-neg10-tuple4/checkpoint7.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16
bash scripts/test_dist_all.sh logs/netVLAD/attention/student/robotcar30k-vgg16/conv5-conv6-vistypepow2-vprLtriplet-attLmse-lvpr1.0-latt1.0-lr0.001-neg10-tuple2/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16
# for pitts-trained netvlad
# bash scripts/test_dist_all.sh logs/netVLAD/attention/teacher/pitts30k-vgg16/conv5-vprlosstriplet-lr0.001-neg10-tuple4/model_best.pth.tar /home/lab/data1/hanjing 512 conv5 vgg16
