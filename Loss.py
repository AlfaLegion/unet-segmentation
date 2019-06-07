import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets.long())

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self,predict,target):
        #print(predict)
        smooth=1
        predictView=predict.view(-1)
        targetView=target.view(-1)
        intersection=(predictView*targetView).sum()
        dice_loss=(2.0*intersection+smooth).div(predictView.sum()+targetView.sum()+smooth)
        return 1-dice_loss
        




