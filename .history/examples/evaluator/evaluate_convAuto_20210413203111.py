import numpy as np
import matplotlib.pyplot as plt

def drawWithDimension(dataset, dimensions, netvlad, alexnet, vgg16, oursA, oursV):
    assert len(dimensions) == len(oursA) == len(oursV)
    netvlad_list = [netvlad]*len(dimensions)
    alexnet_list = [alexnet]*len(dimensions)
    vgg16_list = [vgg16]*len(dimensions)

    plt.figure()
    plt.title('{}'.format(dataset))
    plt.xlabel('Dimension')
    plt.ylabel('AP')
    # plt.grid()
    plt.yticks(np.arange(0.3,1,0.05))
    # plt.xticks(dimensions)
    # plt.xscale('log',base=2)
    plt.plot(dimensions, netvlad_list, linewidth=2, color='yellow', label='netvlad-4096d(auc={:.3f})'.format(netvlad))
    plt.plot(dimensions, alexnet_list, linewidth=2, color='cyan', label='alexnet-272,384d(auc={})'.format(alexnet))
    plt.plot(dimensions, vgg16_list, linewidth=2, color='pink', label='vgg16-614,400d(auc={})'.format(vgg16))
    plt.plot(dimensions, oursA, linewidth=2, color='red', marker='o', label='oursA')
    plt.plot(dimensions, oursV, linewidth=2, color='blue', marker='*', label='oursV')
    
    plt.legend()
    plt.show()
    plt.savefig('./results/{}_dimension.png'.format(dataset))
    plt.close()
