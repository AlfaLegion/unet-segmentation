import torch
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import collections
from PIL import Image
from transform import *
import matplotlib.pyplot as plt
class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        #assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class ToLabel(object):
    def __call__(self, inputs):
        
        
        _,numpyTen=cv2.threshold(inputs,40,1,cv2.THRESH_BINARY)
        tensor= torch.from_numpy(numpyTen)
        tensor.unsqueeze_(0)

        #tensor[tensor!=0]=1 
        #plt.imshow(tensor.numpy())
        #plt.show()
        #aa=tensor[0].numpy()
        #aa=cv2.normalize(numpyTen,None,0,255,cv2.NORM_MINMAX)
        #cv2.imshow("image",aa)
        
        #cv2.waitKey()
        #print(tensor)
        #for i in inputs:
        #    tensors.append(torch.from_numpy(np.array(i)).long())
        return tensor



class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs


class ToSP(object):
    def __init__(self, size):
        self.scale2 = Scale(size/2, Image.NEAREST)
        self.scale4 = Scale(size/4, Image.NEAREST)
        self.scale8 = Scale(size/8, Image.NEAREST)
        self.scale16 = Scale(size/16, Image.NEAREST)
        self.scale32 = Scale(size/32, Image.NEAREST)

    def __call__(self, input):
        input2 = self.scale2(input)
        input4 = self.scale4(input)
        input8 = self.scale8(input)
        input16 = self.scale16(input)
        input32 = self.scale32(input)
        inputs = [input, input2, input4, input8, input16, input32]
        # inputs = [input]

        return inputs
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,img_transofrm=None,label_transform=None,nameImageFile="image",nameMasksFile="mask"):
        self.root_dir = root_dir
        self.img_transofrm = img_transofrm
        self.label_transform = label_transform
        #self.h_flip = HorizontalFlip()
        #self.v_flip = VerticalFlip()
        self.files = list()

        listNamesClass = os.listdir(root_dir)
        print("Found files: ",listNamesClass)
        inex = 0
        for nameClass in listNamesClass:
            pathClass = os.path.join(root_dir,nameClass)
            pathImage = os.path.join(pathClass,nameImageFile)
            pathMask = os.path.join(pathClass,nameMasksFile)

            listImage = os.listdir(pathImage)
            listMask = os.listdir(pathMask)
            lenIms = len(listImage)
            lenMsks = len(listMask)
            if lenIms != lenMsks:
                strError = "error of the dimension of the list of images (" + str(lenIms) + ") and masks (" + str(lenMsks) + ")"
                raise IOError(strError)

            for i in range(lenIms):
                img_file = os.path.join(pathImage,listImage[i])
                msk_file = os.path.join(pathMask,listMask[i])
                self.files.append({
                    "image":img_file,
                    "label":msk_file
                    })

        print("Number of labels: ",len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        datafile = self.files[index]

        #image = cv2.imread(datafile["image"],cv2.IMREAD_COLOR)
        #label = cv2.imread(datafile["label"],cv2.IMREAD_COLOR)
        #label = np.array(label, dtype=np.uint8)

        img_file = datafile["image"]
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img = Image.open(img_file).convert('RGB')

        label_file = datafile["label"]
        label=cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)

        #cv2.imshow("image",img)
        #cv2.imshow("label",label)
        #cv2.waitKey()

        #label = Image.open(label_file)
        #label = np.array(label)

        if self.img_transofrm is not None:
            img_o = self.img_transofrm(img)
            imgs = img_o
        else:
            imgs = img

        if self.label_transform is not None:
            label_o = self.label_transform(label)
            labels = label_o
        else:
            labels = label


        #imgs=torch.from_numpy(np.array(imgs))
        #labels=torch.from_numpy(np.array(labels)).float()
        
        return imgs, labels


if __name__ == "__main__":

    input_transform = Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256),Image.BILINEAR),
        #transforms.ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),

    ])
    target_transform = Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256),Image.NEAREST),
        #ToSP(256),
        #ToLabel(),
        #ReLabel(255, 21),
])

    dataset = SegmentationDataset(r'D:\segmentation\dataset',img_transofrm=input_transform,label_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=True)

    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(imgs.shape)
        print(labels.shape)
        for k in range(10):
            img = imgs[k].numpy()
            print(img.shape)
            #img = np.transpose(img, (1, 2, 0))


            lb = labels[k].numpy()

            #lb = np.transpose(lb, (1, 2, 0))
            aa= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("img",aa)
            cv2.imshow("lb",lb)
            cv2.waitKey()