import torch.nn as nn
import torch
class CrossEntropyLoss2d(nn.Module):
    def __init__(self,ignore_index=255,reduction='elementwise_mean', weight = False):
        super(CrossEntropyLoss2d,self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction    = reduction
        if self.weight:
            # lane and line
            weights_tensor = torch.tensor([0.05, 1.0, 1.0, 1.0, 1.0, 1.0,0.0,0.0])
            # reduce v3
            # weights_tensor = torch.tensor([0.01, 1.0, 1.0, 1.0])
            # weight lossv1
            # weights_tensor = torch.tensor([0.01, 0.95, 1.0, 1.0])
            # weights_tensor = torch.tensor([0.01, 1.0, 1.0, 1.0])
            # reduce v2
            # weights_tensor = torch.tensor([0.01, 0.75, 0.75, 0.75, 1.0])
            # reduce
            # weights_tensor = torch.tensor([0.05, 2.5, 1.0, 1.0, 1.5, 1.5]) #FAIL
            # weights_tensor = torch.tensor([0.05, 1.0, 1.0, 1.0, 1.0, 1.0])
            # all
            # weights_tensor = torch.tensor([0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.loss_fn      = nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction=self.reduction, weight = weights_tensor)
        else:
            self.loss_fn      = nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction=self.reduction)
    def forward(self,pred,target):
        num_classes = pred.size()[1] #19
        pred = pred.transpose(1,2).transpose(2,3).contiguous().view(-1,num_classes) #pred = (2,128,256,19) =view=> (2,,19)
        # print("3",pred.size())
        # print(target.size())
        target = target.view(-1)
        loss = self.loss_fn(pred,target)
        return loss