import torch
import torch.nn as nn

class YMAELoss(nn.Module):

    def __init__(self, eps=1e-9):
        super(YMAELoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        Y_x = (x[:,0,:,:]*0.114) + (x[:,1,:,:]*0.586) + (x[:,2,:,:]*0.299) 
        Y_y = (y[:,0,:,:]*0.114) + (y[:,1,:,:]*0.586) + (y[:,2,:,:]*0.299) 

        diff = Y_x - Y_y

        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss

    def __str__(self):
        return "Y_MAE_LOSS"