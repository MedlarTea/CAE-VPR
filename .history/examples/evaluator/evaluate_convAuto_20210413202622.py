import numpy as np
import matplotlib.pyplot as plt

def drawWithDimension(dataset, dimensions, netvlad, alexnet, vgg16, oursA, oursV):
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
    plt.plot(dimensions, netvlad_list, linewidth=2, color='pink', label='netvlad-4096d(auc={:.3f})'.format(netvlad))
    plt.plot(dimensions, alexnet_list, linewidth=2, color='cyan', label='alexnet-272,384d(auc={})'.format(alexnet))
    plt.plot(dimensions, netvlad_conv5, linewidth=2, color='red', marker='o', label='netvlad-conv5-encoder')
    
    plt.legend()
    plt.show()
    plt.savefig('./results/{}_{}.png'.format(resolution, dataset))
    plt.close()
