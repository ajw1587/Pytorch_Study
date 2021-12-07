# from torch.utils.data import Dataset, DataLoader
# from glob import glob
# import torchvision.transforms as T
# import cv2 as cv
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
#
#
# class custom_dataset(Dataset):
#     def __init__(self, file_path, classes, transform=None):
#         self.image_path = file_path
#         self.label = self.get_label(file_path)
#         # print('np.array(self.label).shape: ', np.array(self.label).shape)
#         self.transform = transform
#         self.classes = classes
#
#     def get_label(self, file_path):
#         label_list = []
#         for path in file_path:
#             sample = path.split('/')[-1]
#             sample = sample.split('\\')[-2]
#             # print(sample)
#             label_list.append(sample)
#         return label_list
#
#     def __len__(self):
#         return len(self.image_path)
#
#     def __getitem__(self, index):
#         img_path = self.image_path[index]
#         img = cv.imread(img_path)
#
#         # print('img.shape: ', img.shape)
#         # print('type(img): ', type(img))
#         # img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
#
#         self.transform = T.Compose([
#             T.ToTensor()]
#         )
#
#         print("img type: ", type(img))
#         # img =
#         img = self.transform(img)
#         # cv.imshow('asdf', img)
#         # cv.waitKey(0)
#
#         # print('changed img shape: ', img.shape)
#         # print('img label: ', self.label[index])
#         return img, self.label[index]
#
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# data_path = glob('./data/CIFAR-10-images-master/train/*/*.jpg')
# test_path = glob('./data/CIFAR-10-images-master/test/*/*.jpg')
#
# train_x = custom_dataset(data_path, classes)
# img, label = train_x[35001]
# print("img, label: ", img.shape, label)
#
# train_loader = DataLoader(custom_dataset(data_path, classes),
#                           batch_size=1,
#                           shuffle=True)
# test_loader = DataLoader(custom_dataset(test_path, classes),
#                          batch_size=1,
#                          shuffle=True)
########################################################################
# from torch.utils.data import Dataset, DataLoader
# from glob import glob
# import torchvision.transforms as T
# import cv2 as cv
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#
# class custom_dataset1(Dataset):
#     def __init__(self, file_path, my_classes, transforms=None):
#         self.image_path = file_path
#         self.label = self.get_label(file_path)
#         self.classes = my_classes
#         self.transforms = transforms
#
#     def get_label(self, file_path):
#         label_list = []
#         for file in file_path:
#             temp = file.split('/')[-1]
#             temp = temp.split('\\')[-2]
#             label_list.append(temp)
#         return label_list
#
#     def __len__(self):
#         return len(self.image_path)
#
#     def __getitem__(self, idx):
#         img = cv.imread(self.image_path[idx])
#
#         if self.transforms is not None:
#             return img, self.label[idx]
#         else:
#             return self.tranforms(img), self.label[idx]
#
#
# train_data_path = glob('./data/CIFAR-10-images-master/train/*/*.jpg')
# test_data_path = glob('./data/CIFAR-10-images-master/test/*/*.jpg')
# my_mean = [0.485, 0.456, 0.406]
# my_std = [0.229, 0.224, 0.225]
#
# my_transforms = T.Compose([
#     T.Resize((100, 100)),
#     T.Normalize(mean=my_mean, std=my_std)
# ])
#
# train_loader = DataLoader(custom_dataset(train_data_path, classes, my_transforms),
#                           batch_size=10,
#                           shuffle=True,
#                           drop_last=True)
# test_loader = DataLoader(custom_dataset(test_data_path, classes, my_transforms),
#                          batch_size=10,
#                          shuffle=True,
#                          drop_last=True)
#
# print(train_loader)
#
# import pandas as pd
#
#
# # class my_csv_dataset(Dataset):
# #     def __init__(self, file_path, classes, transforms):
# #         self.file_path = pd.read_csv(file_path)
# #         self.label
# #         self.classes
# #         self.transforms = transforms
# #
# #     def __len__(self):
# #
# #     def __getitem__(self):
#
# test = pd.read_csv('./data/test_csv_file.csv')
# print(test.iloc[0, 0])

from glob import glob
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import torchvision.transforms as T

# class name
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# data path
train_data = glob('./data/CIFAR-10-images-master/train/*/*.jpg')
test_data = glob('./data/CIFAR-10-images-master/test/*/*.jpg')

# normalize parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class my_dataset(Dataset):
    def __init__(self, image_path, classes, transform):
        self.image_path = image_path
        self.labels = self.get_labels(image_path)
        self.transform = transform
        self.classes = classes

    def get_labels(self, image_path):
        label_list = []
        for path in image_path:
            temp = image_path.split('/')[-1]
            temp = temp.split('\\')[-2]
            label_list.append(temp)
        return label_list

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = cv.imread(self.image_path[idx])

        if self.transform is not None:
            return img, self.label[idx]
        else:
            return self.transform(img), self.label[idx]


# my_transforms = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=mean, std=std)
# ])
#
# train_data_loader = DataLoader(my_dataset(train_data, classes, my_transforms),
#                                batch_size=10,
#                                shuffle=True,
#                                drop_last=True)
# test_data_loader = DataLoader(my_dataset(test_data, classes, my_transforms),
#                               batch_size=10,
#                               shuffle=True,
#                               drop_last=True)

