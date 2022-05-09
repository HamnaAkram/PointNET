from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from util import default_transforms, read_off
from check_data import read_off
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        folders=[]
        self.root_dir = root_dir
        for fol in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir,fol)):
                folders.append(fol)
        folders = sorted(folders)
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = os.path.join(root_dir,category,folder)
            for file in os.listdir(new_dir):
                    if file.endswith('.off'):
                        sample = {}
                        sample['pcd_path'] = os.path.join(new_dir,file)
                        sample['category'] = category
                        self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        #with open(pcd_path, 'r') as f:
        pointcloud = self.__preproc__(pcd_path)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}