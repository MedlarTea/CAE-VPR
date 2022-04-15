import numpy as np
import matplotlib.pyplot as plt

def drawWithDimension(resolution, dataset, original, dimensions, netvlad_original, netvlad_conv5_original, netvlad_conv5):
    netvlad_original_list = [netvlad_original]*len(dimensions)
    netvlad_conv5_original_list = [netvlad_conv5_original]*len(dimensions)
    plt.figure()
    plt.title('{} {}'.format(resolution, dataset))
    plt.xlabel('encoder dimension')
    plt.ylabel('auc')
    # plt.grid()
    plt.yticks(np.arange(0.3,1,0.05))
    # plt.xticks(dimensions)
    # plt.xscale('log',base=2)
    plt.plot(dimensions, netvlad_original_list, linewidth=2, color='pink', label='netvlad-4096,auc={}'.format(netvlad_original))
    plt.plot(dimensions, netvlad_conv5_original_list, linewidth=2, color='cyan', label='netvlad-conv5-{},auc={}'.format(original,netvlad_conv5_original))
    plt.plot(dimensions, netvlad_conv5, linewidth=2, color='red', marker='o', label='netvlad-conv5-encoder')
    
    plt.legend()
    plt.show()
    plt.savefig('./results/{}_{}.png'.format(resolution, dataset))
    plt.close()
