import os
import numpy as np
import random
from PIL import Image

def get_files_from_directory(directory_path):
    """
    Args:
        directory_path(str): path that stored the images
    Output:
        files(List): absolute paths of all images
    """
    files = []
    for x in os.listdir(directory_path):
        if (x.endswith("png")) and x.find("entropy")==-1 and x.find("patches")==-1:
            # print(x)
            files.append(os.path.join(directory_path, x))
    return files

if __name__ =="__main__":
    image_files = get_files_from_directory("./")
    
    for image_path in image_files:
        image = np.array(Image.open(image_path), dtype=np.uint8)
        row_sizes = [32, 48, 96, 120, 240]
        col_sizes = [64, 80, 128, 160, 320]
        
        for row_size in row_sizes:
            for col_size in col_sizes:
                patches = np.zeros((480, 640, 3), dtype=np.uint8)
                print(image_path)
                patches_path = image_path.rstrip(".png") + "_{}_{}_{}_patches.png".format(row_size, col_size, row_size*col_size)

                row_start = random.randint(0, 480-row_size)
                col_start = random.randint(0, 640-col_size)
                patch = image[row_start:row_start+row_size, col_start:col_start+col_size, :]  # (row_size, col_size, 3)
                rows = int(patches.shape[0]/patch.shape[0])
                cols = int(patches.shape[1]/patch.shape[1])
                print(rows, cols)
                for i in range(rows):
                    for j in range(cols):
                        patches[i*row_size: (i+1)*row_size, j*col_size: (j+1)*col_size, :] = patch
                patches = Image.fromarray(patches)
                patches.save(patches_path)