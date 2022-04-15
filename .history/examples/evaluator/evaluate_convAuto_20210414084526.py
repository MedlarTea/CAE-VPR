import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

matches = {
    "robotcar(dbNight vs. qSnow)": ["robotcar_qSnow_dbNight_distri_trueMatches.npy", "robotcar_qSnow_dbNight_distri_falseMatches.npy"]
}
def drawWithDimension(dataset, dimensions, netvlad, alexnet, vgg16, oursA, oursV):
    assert len(dimensions) == len(oursA) == len(oursV)
    netvlad_list = [netvlad]*len(dimensions)
    alexnet_list = [alexnet]*len(dimensions)
    vgg16_list = [vgg16]*len(dimensions)

    plt.figure()
    plt.title('{}'.format(dataset), weight='bold')
    plt.xlabel('Dimension')
    plt.ylabel('AP(Average Precision)')
    # plt.grid()
    plt.yticks(np.arange(0.2,1,0.05))
    # plt.xticks(dimensions)
    # plt.xscale('log',base=2)
    plt.plot(dimensions, netvlad_list, linewidth=2, color='yellow', label='Netvlad-4096d(AP={:.3f})'.format(netvlad))
    plt.plot(dimensions, alexnet_list, linewidth=2, color='pink', label='Alexnet-272,384d(AP={:.3f})'.format(alexnet))
    plt.plot(dimensions, vgg16_list, linewidth=2, color='cyan', label='Vgg16-614,400d(AP={:.3f})'.format(vgg16))
    plt.plot(dimensions, oursA, linewidth=2, color='red', marker='o', label='oursA')
    plt.plot(dimensions, oursV, linewidth=2, color='blue', marker='*', label='oursV')

    plt.legend()
    plt.show()
    plt.savefig('./results/{}_dimension.png'.format(dataset), dpi=600)
    plt.close()

def drawL2Distribution(method, dataset, trueMatches, falseMatches):
    trueMatches = np.load(trueMatches)
    falseMatches = np.load(falseMatches)
    true_mean = np.mean(trueMatches)
    true_std = np.var(trueMatches)
    false_mean = np.mean(falseMatches)
    false_std = np.var(falseMatches)

    plt.figure()
    counts_x, bins_x = np.histogram(trueMatches,bins=15)
    plt.hist(bins_x[:-1], bins_x, weights=counts_x/len(trueMatches), color="darkorange", edgecolor = 'black', alpha=0.5, label="True matches(u={:.3f}, var={:.3f})".format(true_mean, true_std))
    counts_y, bins_y = np.histogram(falseMatches,bins=15)
    plt.hist(bins_y[:-1], bins_y, weights=counts_y/len(falseMatches), color="darkcyan", edgecolor = 'black', alpha=0.5, label="False matches(u={:.3f}, var={:.3f})".format(false_mean, false_std))
    plt.legend()
    plt.xlabel('L2 Distance')
    plt.ylabel('Probability')
    plt.title('{}'.format(dataset), weight='bold')
    plt.savefig("./results/{}_{}_l2Distri.png".format(dataset, method),dpi=600)
    plt.close()


if __name__ == '__main__':
    # draw the dimension
    # for synth.Norland
    dataset = 'Synth. Norland'
    dimensions = ['256', '512', '1024', '2048', '4096', '8192', '16384']
    netvlad = 0.402
    alexnet = 0.956
    vgg16 = 0.815
    oursA = [0.924, 0.940, 0.958, 0.974, 0.966, 0.963, 0.964]
    oursV = [0.856, 0.920, 0.958, 0.964, 0.972, 0.974, 0.974]
    drawWithDimension(dataset, dimensions, netvlad, alexnet, vgg16, oursA, oursV)

    # for robotcar(dbNight vs. qSnow)
    dataset = 'robotcar(dbNight vs. qSnow)'
    dimensions = ['256', '512', '1024', '2048', '4096', '8192', '16384']
    netvlad = 0.893
    alexnet = 0.657
    vgg16 = 0.810
    oursA = [0.503, 0.812, 0.916, 0.945, 0.968, 0.977, 0.979]
    oursV = [0.896, 0.914, 0.948, 0.964, 0.972, 0.980, 0.984]
    drawWithDimension(dataset, dimensions, netvlad, alexnet, vgg16, oursA, oursV)

    # draw the l2 distribution
    # for robotcar(dbNight vs. qSnow)
    dataset = "robotcar(dbNight vs. qSnow)"
    root_dir = "/home/lab/data1/hanjingModel/OpenIBL/logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-dimension4096"
    trueMatches = osp.join(root_dir, matches[dataset][0])
    falseMatches = osp.join(root_dir, matches[dataset][1])
    method = "vggConAuto_4096"
    drawL2Distribution(method, dataset, trueMatches, falseMatches)

    root_dir = "/home/lab/data1/hanjingModel/OpenIBL/logs/vgg16"
    trueMatches = osp.join(root_dir, matches[dataset][0])
    falseMatches = osp.join(root_dir, matches[dataset][1])
    method = "vgg16"
    drawL2Distribution(method, dataset, trueMatches, falseMatches)

