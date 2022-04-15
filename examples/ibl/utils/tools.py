import os
import torch
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from scipy.spatial.distance import cdist,cosine
import scipy.io as scio
import cv2
import faiss

def input_transform():
    return transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                   std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
    ])

class pitts(data.Dataset):
    def __init__(self, image_path_list, input_transform=input_transform()):
        super().__init__()
        self.input_transform = input_transform
        self.image_path_list = image_path_list

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index][0]).convert('RGB')
        # semantic = Image.open(self.image_path_list[index][1])
        if self.input_transform:
            img = self.input_transform(img)
        # semantic_mask = self.preprocess_semantic(semantic)
        # img = torch.cat((img, semantic_mask), 0)
        return img

    def __len__(self):
        return len(self.image_path_list)

    def preprocess_semantic(self, semantic_mask):
        r'''Preprocess the semantic mask

            The semantic mask is inferred from deeplabv3plus trained in cityscapes. Then transform [0,18] of uint8 to [0,1] of torch.float32  
            Input:
                PIL.Image size(W,H)/(1,W,H) in range[0,18]
                with 0-road, 1-sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-light, 7-traffic-sign, 8-vegetation, 9-terrain, 10-sky, 11-18-dynamic objects
            Output:
                (1,H,W) of torch.float32 in range[0,1]
        '''
        return (transforms.ToTensor()(semantic_mask)+(1/255)) * 255 / 19

class conv_auto_dataset(data.Dataset):
    def __init__(self, image_path_list, semantic_path_list=None, input_transform=input_transform()):
        super().__init__()
        self.input_transform = input_transform
        self.image_path_list = image_path_list
        self.semantic_path_list = semantic_path_list
        self.average_rgb = np.load('./pretrained_models/average_rgb.npy')

    def __getitem__(self, index):
        # if self.semantic_path_list == None:
        #     img = Image.open(self.image_path_list[index])
        #     img = np.array(img) - self.average_rgb
        #     img = cv2.resize(img, (640,480))
        #     img = np.array(img, dtype=np.float32)
        #     img = transforms.ToTensor()(img)
        #     return img
        # else:
        #     assert self.image_path_list[index].split('/')[-1].split('.')[0] == self.semantic_path_list[index].split('/')[-1].split('.')[0]
        #     img = Image.open(self.image_path_list[index])
        #     img = np.array(img) - self.average_rgb
        #     img = cv2.resize(img, (640,480))
        #     img = np.array(img, dtype=np.float32)
        #     # semantic
        #     semantic = np.load(self.semantic_path_list[index])  # (1,480,640)
        #     semantic = np.tile(np.expand_dims(np.squeeze(semantic, 0), 2), 3).astype(np.uint8)  # (480,640,3)
        #     semantic = cv2.resize(semantic, (640,480))
        #     semantic_mask = np.ones((480, 640, 3),dtype=np.float32)
        #     # remove dynamic, 10-sky
        #     for i  in [11, 12, 13, 14, 15, 16, 17, 18]:
        #         semantic_mask[semantic==i] = 0
        #     gt = img * semantic_mask
        #     img = transforms.ToTensor()(img)
        #     gt = transforms.ToTensor()(gt)
        #     return img, gt
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_path_list)

def load_gt(_database):
    datatype = _database.split('/')[-2]
    gt = scio.loadmat(_database)
    gt = gt[datatype]
    gt = np.array(gt, dtype=np.uint)
    # print("original shape: {}".format(gt.shape))
    if datatype == 'UAcampus':
        img_nums = 647
        gt = gt[:img_nums, :img_nums]
    elif datatype == 'Mapillary':
        img_nums = 1301
        gt = gt[:img_nums, :img_nums]
    # print("After cropped, the shape: {}".format(gt.shape))
    gt_list=[]  # (img_nums, [1,2,3,...])
    for i in range(img_nums):
        gt_list.append(np.where(gt[i]==1)[0])
    return gt_list

def cal_UAcampus_matches(database_file, database, query):
    true_neigh_list = load_gt(database_file)
    # use faiss来加速匹配
    print('====> Building uac/map faiss index')
    pool_size = query.shape[1]
    faiss.normalize_L2(query)
    faiss.normalize_L2(database)
    index=faiss.IndexFlatIP(pool_size)
    index.train(database)
    index.add(database)
    distance, predictions = index.search(query, 1)  # 找最近邻  distance-shape[n,1] predictions-shape[n,1]
    matchesTop = np.concatenate((predictions, distance), axis=1)  # shape[n,2]
    # print(matchesTop)

    # topN = 10
    # distMat = cdist(database, query, "cosine")  # 计算距离矩阵
    # matchesTopN = np.argsort(distMat, axis=0)[:topN, :]  # 小在前,按列排序
    # matches_dis = np.sort(distMat, axis=0)[0, :]  # 小在前,按列排序
    # matchesTop = np.vstack((matchesTopN[0, :], matches_dis)).transpose() # (n,2)--(n,[id, score])
    # save the match file
    return matchesTop, true_neigh_list

def load_trainingSet():
    image_list = []
    semantic_list = []
    image_path = "/home/lab/data1/robotcar"
    image_date_list = sorted(os.listdir(image_path))
    #存储描述符的路径
    for date in image_date_list:
            image_dirname1 = os.path.join(image_path, date)
            image_view_list = os.listdir(image_dirname1)
            # desc_path = mkdir(base_path, date)
            # if "stereo" in image_view_list and date in ['2014-12-10-18-10-50', '2015-11-13-10-28-08']:
            if "stereo" in image_view_list and date in ['2015-11-13-10-28-08']:
                image_dirname2 = os.path.join(image_dirname1, "image_front_2m")
                semantic_image2 = os.path.join(image_dirname1, "semantic_images/OCRNet")
                basename = sorted(os.listdir(image_dirname2))
                img_path_list = list(map(lambda x:os.path.join(image_dirname2, x), basename))[900:]  # 与测试集分离
                semantic_path_list = list(map(lambda x:os.path.join(semantic_image2, x.split('.')[0]+'.npy'), basename))[900:]  # 与测试集分离
                image_list = image_list + img_path_list
                semantic_list = semantic_list + semantic_path_list
    print("Total images: {}".format(len(image_list)))
    return image_list

def load_semantic_trainingSet():
    semantic_list = []
    image_path = "/home/lab/data1/robotcar"
    image_date_list = sorted(os.listdir(image_path))
    #存储描述符的路径
    for date in image_date_list:
            image_dirname1 = os.path.join(image_path, date)
            image_view_list = os.listdir(image_dirname1)
            # desc_path = mkdir(base_path, date)
            # if "stereo" in image_view_list and date in ['2014-12-10-18-10-50', '2015-11-13-10-28-08']:
            if "stereo" in image_view_list and date in ['2015-11-13-10-28-08']:
                semantic_image2 = os.path.join(image_dirname1, "semantic_images/OCRNet/npy")
                basename = sorted(os.listdir(semantic_image2))
                semantic_path_list = list(map(lambda x:os.path.join(semantic_image2, x), basename))[900:]  # 与测试集分离
                semantic_list = semantic_list + semantic_path_list
    print("Total semantic_images: {}".format(len(semantic_list)))
    return semantic_list

def load_imagePaths(image_path_file):
    images = sorted(os.listdir(image_path_file))
    img_path_list = list(map(lambda x:os.path.join(image_path_file, x), images))
    return img_path_list

def load_uac_imagePaths(image_path_file):
    length = len(os.listdir(image_path_file))
    img_path_list = []
    for i in range(length):
        img_path_list.append(os.path.join(image_path_file, str(i)+'.jpg'))
    return img_path_list

def load_uac_semanticPaths(semantic_path_file):
    length = len(os.listdir(semantic_path_file))
    semantic_path_list = []
    for i in range(length):
        semantic_path_list.append(os.path.join(semantic_path_file, str(i)+'.npy'))
    return semantic_path_list

def load_testingSet():
    robotcar_base_dir = '/home/lab/data/yehanjingModel/LoopClosure/fusion/test-fusion'
    # query_Robotcar_front_autumn_file = os.path.join(robotcar_base_dir, 'overcast-autumn/2014-11-28-12-07-13', 'image_front_2m')
    query_Robotcar_front_night_file = os.path.join(robotcar_base_dir, 'night-autumn/2014-12-10-18-10-50', 'image_front_2m')
    # query_Robotcar_front_summer_file = os.path.join(robotcar_base_dir, 'overcast-summer/2015-05-19-14-06-38', 'image_front_2m')
    # query_Robotcar_front_winter_file = os.path.join(robotcar_base_dir, 'overcast-winter/2015-02-03-08-45-10', 'image_front_2m')
    refer_Robotcar_front_autumn_file = os.path.join(robotcar_base_dir, 'overcast-autumn/2015-11-13-10-28-08', 'image_front_2m')
    # refer_Robotcar_rear_autumn_file = os.path.join(robotcar_base_dir, 'overcast-autumn/2015-11-13-10-28-08', 'image_rear_2m')

    other_base_dir = '/home/lab/data/yehanjingModel/LoopClosure/dataset'
    query_UAcampus_file = os.path.join(other_base_dir, 'UAcampus/UAcampus_query')
    refer_UAcampus_file = os.path.join(other_base_dir, 'UAcampus/UAcampus_train')
    query_Mapillary_file = os.path.join(other_base_dir, 'Mapillary/Mapillary_query')
    refer_Mapillary_file = os.path.join(other_base_dir, 'Mapillary/Mapillary_train')
    query_Mapillary_semantic_file = os.path.join(other_base_dir, 'Mapillary/semantic_images/OCRNet/query/npy')
    refer_Mapillary_semantic_file = os.path.join(other_base_dir, 'Mapillary/semantic_images/OCRNet/train/npy')
    # return (load_imagePaths(query_Robotcar_front_autumn_file), load_imagePaths(query_Robotcar_front_night_file), 
    #         load_imagePaths(query_Robotcar_front_summer_file), load_imagePaths(query_Robotcar_front_winter_file),
    #         load_imagePaths(refer_Robotcar_front_autumn_file), load_imagePaths(refer_Robotcar_rear_autumn_file),
    #         load_uac_imagePaths(query_UAcampus_file), load_uac_imagePaths(refer_UAcampus_file),
    #         load_uac_imagePaths(query_Mapillary_file), load_uac_imagePaths(refer_Mapillary_file))

    return (load_imagePaths(query_Robotcar_front_night_file), load_imagePaths(refer_Robotcar_front_autumn_file),
            load_uac_imagePaths(query_UAcampus_file), load_uac_imagePaths(refer_UAcampus_file),
            load_uac_imagePaths(query_Mapillary_file), load_uac_imagePaths(refer_Mapillary_file),
            load_uac_semanticPaths(query_Mapillary_semantic_file), load_uac_semanticPaths(refer_Mapillary_semantic_file))

def load_pitts250k_trainingSet(image_root, semantic_root):
    image_list = []
    semantic_root = semantic_root
    image_root = image_root
    image_date_list = sorted(os.listdir(image_root))  # [000,001,...,010]
    #存储描述符的路径
    for date in image_date_list:
            image_dirname = os.path.join(image_root, date)
            semantic_dirname = os.path.join(semantic_root, date)
            images = os.listdir(image_dirname)  # [000713_pitch2_yaw3.jpg,...]
            # desc_path = mkdir(base_path, date)
            # if "stereo" in image_view_list and date in ['2014-12-10-18-10-50', '2015-11-13-10-28-08']:
            for image in images:
                image_path = os.path.join(image_dirname, image)
                semantic_path = os.path.join(semantic_dirname, image.split('.')[0]+'_mask.png')
                image_list.append([image_path, semantic_path])
    return image_list

def test_others(model, pca, output_dim, query_set, refer_set, dType, datatype=None, index=None, query_semantic=None, refer_semantic=None):
    nums = len(query_set)
    whole_set = query_set + refer_set
    if query_semantic == None:
        whole_semantic = None
    else:
        whole_semantic = query_semantic + refer_semantic
    whole_set = tools.conv_auto_dataset(whole_set, whole_semantic)

    test_data_loader = DataLoader(dataset=whole_set, 
                num_workers=8, batch_size=32, shuffle=False, 
                pin_memory=True)
    pca.load(gpu=cuda)

    model.eval()
    with torch.no_grad():
        # print('====> Extracting Features')
        beg=0
        end=0
        dbFeat = np.empty((len(whole_set), output_dim))
        for input in test_data_loader:
            # print(input.shape)
            input = input.cuda()
            image_encoding = model.encoder(input)
            encoded = model.pool(image_encoding)
            encoded = pca.infer(encoded)
            encoded = encoded.cpu().numpy()
            end+=encoded.shape[0]
            dbFeat[beg:end, :] = encoded
            beg=end
            del input, image_encoding, encoded
    del test_data_loader
        # extracted for both db and query, now split in own sets
    qFeat = dbFeat[:nums,:].astype('float32')
    dbFeat = dbFeat[nums:,:].astype('float32')
    if dType == 'robotcar':
        return evaluate(qFeat, dbFeat, datatype, index)
    elif dType == 'uac':
        refer_uac = '/home/lab/data/yehanjingModel/LoopClosure/dataset/UAcampus/UAcampusGroundTruth.mat'
        return evaluate_UAC(refer_uac, dbFeat, qFeat)
    elif dType == 'map':
        refer_map = '/home/lab/data/yehanjingModel/LoopClosure/dataset/Mapillary/MapillaryGroundTruth.mat'
        return evaluate_UAC(refer_map, dbFeat, qFeat)
    else:
        raise ValueError('Unknown dType')

