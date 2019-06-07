import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchsummary import summary
import Loss
import DataLoader
from torch.autograd import Variable
import torchvision.utils as v_utils
from PIL import Image
from UnetBlocks import*

class UNeTT(nn.Module):
    def __init__(self,input_dim,out_dim,start_channels):
        super(UNeTT,self).__init__()

        self.downBlock_1 = ConvBlock(input_dim,start_channels,1)
        self.mp_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.downBlock_2 = ConvBlock(start_channels,start_channels * 2,1)
        self.mp_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.downBlock_3 = ConvBlock(start_channels * 2,start_channels * 4,1)
        self.mp_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.downBlock_4 = ConvBlock(start_channels * 4,start_channels * 8,1)
        self.mp_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bottleneck = Bottleneck(start_channels * 8,start_channels * 16,1)

        self.deconvBlock_1 = DeconvBnRelu(start_channels * 16,start_channels * 8,1,1)
        self.convBlock_1 = ConvBlock(start_channels * 16,start_channels * 8,1)
       

        self.deconvBlock_2 = DeconvBnRelu(start_channels * 8,start_channels * 4,1,1)
        self.convBlock_2 = ConvBlock(start_channels * 8,start_channels * 4,1)


        self.deconvBlock_3 = DeconvBnRelu(start_channels * 4,start_channels * 2,1,1)
        self.convBlock_3 = ConvBlock(start_channels * 4,start_channels * 2,1)

        self.deconvBlock_4 = DeconvBnRelu(start_channels * 2,start_channels,1,1)
        self.convBlock_4 = ConvBlock(start_channels * 2,start_channels,1)

        self.outConv = nn.Conv2d(start_channels,out_dim,1,stride=1,padding=0)
        self.act_sigmoid_out = nn.Sigmoid()

    def forward(self,input):

        down_1 = self.downBlock_1(input)
        pool_1 = self.mp_1(down_1)

        down_2 = self.downBlock_2(pool_1)
        pool_2 = self.mp_2(down_2)

        down_3 = self.downBlock_3(pool_2)
        pool_3 = self.mp_3(down_3)

        down_4 = self.downBlock_4(pool_3)
        pool_4 = self.mp_4(down_4)

        bottleneck = self.bottleneck(pool_4)

        decon_1 = self.deconvBlock_1(bottleneck)
        concat_1 = torch.cat([decon_1,down_4],dim=1)
        up_1 = self.convBlock_1(concat_1)
        

        decon_2 = self.deconvBlock_2(up_1)
        concat_2 = torch.cat([decon_2,down_3],dim =1)
        up_2 = self.convBlock_2(concat_2)

        decon_3 = self.deconvBlock_3(up_2)
        concat_3 = torch.cat([decon_3,down_2],dim =1)
        up_3 = self.convBlock_3(concat_3)

        decon_4 = self.deconvBlock_4(up_3)
        concat_4 = torch.cat([decon_4,down_1],dim =1)
        up_4 = self.convBlock_4(concat_4)

        output = self.outConv(up_4)
        output = self.act_sigmoid_out(output)

        return output


def unet():

    model = UNeTT(3,1,64)
    model = model.float().cuda()
    model.eval()
    #example = torch.rand(1, 1, 250, 200).cuda()
    #out = model(example)
    #print(out.shape)
    summary(model,(3, 256, 256))

def train():
        
    batch_size = 32
    img_size = 256
    lr = 0.007
    epoch = 90
    
    img_dir = "D:\\segmentation\\dataset_two_classes\\"

    input_transform = transforms.Compose([#transforms.ToPILImage(),
        #transforms.Resize((img_size, img_size),Image.BILINEAR),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    target_transform = transforms.Compose([#transforms.ToPILImage(),
        #transforms.Resize((img_size, img_size),Image.BILINEAR),
        DataLoader.ToLabel()])
    img_data = DataLoader.SegmentationDataset(img_dir,img_transofrm=input_transform,label_transform=target_transform)
    img_batch = torch.utils.data.DataLoader(img_data,batch_size=batch_size,shuffle=True,num_workers=0)

    print("\nInitialization net...")
    generator = UNeTT(3,1,16).cuda()
    print("Complete")

    #recon_loss_func = nn.MSELoss()
    #recon_loss_func = Losses.CrossEntropyLoss2d()
    #recon_loss_func=nn.NLLLoss()
    recon_loss_func = Losses.DiceLoss()
    #recon_loss_func=nn.CrossEntropyLoss()
    #recon_loss_func=nn.BCEWithLogitsLoss()
    
    gen_optimizer = torch.optim.SGD(generator.parameters(),lr=lr,momentum=0.6,weight_decay=0.0005)
    
    
    generator.train()
    
    print("\nTrain...\n")
    for i in range(epoch):
        print("Epoch: ",i + 1,"/",epoch)
        avg_loss = 0.0
        num = 0
        for _,(images,labels) in enumerate(img_batch):
            
            
            ########################################
            #import cv2
            #import numpy as np
            #a=images[0].numpy().copy()
            #a=a.transpose((2,1,0)).copy()
            
            #a=a.transpose((1,0,2))
            #a=cv2.normalize(a,None,0.,255.,cv2.NORM_MINMAX)
            #cv2.imshow("image",a.astype(dtype=np.uint8))

            #b=labels[0].numpy().copy()
            #b=b.transpose((2,1,0)).copy()
            #b=b.transpose((1,0,2),)
            #b=cv2.normalize(b,None,0,255,cv2.NORM_MINMAX)

            #cv2.imshow("label",b.astype(dtype=np.uint8))
            #cv2.waitKey()
            ########################################

            gen_optimizer.zero_grad()
            
            input = Variable(images).cuda(0)
            labels = Variable(labels).cuda(0).float()
    
            #import sys
            #np.set_printoptions(threshold=sys.maxsize)
            #print(y_[0].cpu().numpy())
    
            output = generator(input)
            
            #y_flat = y.view(-1)
            #y_flat_ = y_.view(-1)
            #print(y_flat)
            #print(y_flat_)
            loss = recon_loss_func(output,labels)
            avg_loss+=loss.item()
            num+=1
            loss.backward()
            gen_optimizer.step()
            if _ % 500 == 0:
                 #print("loss: ",loss.item())
 
                 v_utils.save_image(input.cpu().data,"D:\\test\\work_temp\\work_temp\\result\\original_image_{}_{}.png".format(i,_))
                 v_utils.save_image(labels.cpu().data,"D:\\test\\work_temp\\work_temp\\result\\label_image_{}_{}.png".format(i,_))
                 #output = nn.functional.sigmoid(output)
                 v_utils.save_image(output.cpu().data,"D:\\test\\work_temp\\work_temp\\result\\gen_image_{}_{}.png".format(i,_))
                 #print(output.cpu())
        
        print("number of iterations per epoch: ",num)    
        print("avg loss: ",avg_loss/num,"\n")
        

    SavePathModel = "SegModelDCandRasp_1.pt"

    torch.save(generator,SavePathModel)

def convertModel(SavePathModel,newNameModel):
    print("Start...")
    device = torch.device('cpu')
    model = torch.load(SavePathModel,map_location=device)
    model = model.float()
    model.eval()
    example = torch.rand(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(newNameModel)
    print("Complete")

if __name__ == "__main__":
    train()

    #convertModel("SegModelDC_3.pt","SegDCAcript_3.pt")









