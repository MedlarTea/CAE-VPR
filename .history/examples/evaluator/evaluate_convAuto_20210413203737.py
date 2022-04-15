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
    plt.savefig('./results/{}_dimension.png'.format(dataset), dpi=95)
    plt.close()


if __name__ == '__main__':
    # draw the dimension
    # for synth.Norland
    dataset = 'Synth. Norland'
    dimensions = ['256', '512', '1024', '2048', '4096', '8192', '16384']
    netvlad = 0.402
    alexnet = 0.956
    vgg16 = 0.815
    oursA = []
    oursV = [0.856, 0.920, 0.958, 0.964, 0.972, 0.974, 0.974]
