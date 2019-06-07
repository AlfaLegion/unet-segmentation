import torch
from torchvision import transforms
import os
import numpy as np
import cv2
import time as t


def SegmentationEval():
    

    map_dev = "cpu"
    device = torch.device(map_dev)
    isWriteResult = False
    #dirSaveResult="D:\\tasks\\dogs-vs-cats\\"
    pathModel = "SegDCAcript_2.pt"
    model = torch.jit.load(pathModel,map_dev)
    pathDataInput="D:\\segmentation\\TestDataset\\hammer\\image\\"
    #pathDataInput="D:\\segmentation\\small_dest\\hammer\\image\\"
    #pathDataInput = "D:\\segmentation\\datasetResized\\hammer\\image\\"
    #pathDataInput = "D:\\segmentation\\TestDataset\\hammerVal\\image\\"
    #pathDataInput = "D:\\segmentation\\datasetNewBackground\\train\\hammer\\image\\"
    listImg = os.listdir(pathDataInput)

    
    for name in listImg:
       
       if name.find(".jpg")<0:
           continue

       img = cv2.imread(pathDataInput + name,cv2.IMREAD_COLOR)
       img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       cloneMat = cv2.resize(img,(256,256),None,0,0,cv2.INTER_AREA)
       input = transforms.functional.to_tensor(cloneMat)
       input.unsqueeze_(0)

       start_time = t.time()
       input = input.to(device)
       out = model(input)
       finish_time = t.time()
       print(name)
       print("time: ",finish_time-start_time)
       #print(out.shape)

       #print(type(out))
       mask = out.detach().cpu().numpy()[0]
       mask = np.transpose(mask,(1,2,0))
       #print(mask)
       norm_image = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

       _,norm_image = cv2.threshold(norm_image,100,255,cv2.THRESH_BINARY)

       norm_image=cv2.cvtColor(norm_image,cv2.COLOR_GRAY2RGB)
       norm_image[:,:,2]=255

       print(norm_image.shape)
       print(cloneMat.shape)
       qwerty=cv2.addWeighted(cloneMat,0.6,norm_image,0.4,0,None,cv2.CV_8UC3)
       cv2.imshow("qwerty",qwerty)
       cv2.imshow("norm_image",norm_image)
       cv2.imshow("cloneMat",cloneMat)
       cv2.waitKey()

def SegmentationVidosik():


     map_dev = "cuda"
     device = torch.device(map_dev)
     pathModel = "SegDCAcript_3.pt"
     model = torch.jit.load(pathModel,map_dev)
     pathVidosik="test_video_5.avi"
     Videocap=cv2.VideoCapture(pathVidosik)
     while(True):
         r,frame=Videocap.read()
         if not r:
             break
         frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
         cloneMat = cv2.resize(frame,(256,256),None,0,0,cv2.INTER_AREA)

         input = transforms.functional.to_tensor(cloneMat)
         input.unsqueeze_(0)

         input = input.to(device)
         out = model(input)
         mask = out.detach().cpu().numpy()[0]
         mask = np.transpose(mask,(1,2,0))
         norm_image = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

         _,norm_image = cv2.threshold(norm_image,100,255,cv2.THRESH_BINARY)
         
         norm_image=cv2.cvtColor(norm_image,cv2.COLOR_GRAY2RGB)
         norm_image[:,:,2]=255
         qwerty=cv2.addWeighted(cloneMat,0.6,norm_image,0.4,0,None,cv2.CV_8UC3)

         cv2.imshow("qwerty",qwerty)
         cv2.imshow("norm_image",norm_image)
         cv2.imshow("cloneMat",cloneMat)
         cv2.waitKey(33)






#SegmentationVidosik()
SegmentationEval()



