import torch
import torch.nn as nn
import torch.nn.functional as F

class JacintoNet(nn.Module):
    def __init__(self, in_depth, num_classes, input_size, phase = 'train'):
        super(JacintoNet, self).__init__()
        self.in_depth = in_depth
        self.num_class = num_classes
        self.input_size = input_size
        self.phase = phase

        self.conv1a = self._make_conv_block(self.in_depth, 32, 5, 2, 2, 1, 1)
        self.conv1b = self._make_conv_block(32, 32, 3, 1, 1, 1, 4) 
        
        self.res2a_branch2a = self._make_conv_block(32, 64, 3, 1, 1, 1, 1) 
        self.res2a_branch2b = self._make_conv_block(64, 64, 3, 1, 1, 1, 4) 
        
        self.res3a_branch2a = self._make_conv_block(64, 128, 3, 1, 1, 1, 1) 
        self.res3a_branch2b = self._make_conv_block(128, 128, 3, 1, 1, 1, 4)
        
        self.res4a_branch2a = self._make_conv_block(128, 256, 3, 1, 1, 1, 1) 
        self.res4a_branch2b = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        
        self.res5a_branch2a = self._make_conv_block(256, 512, 3, 1, 2, 2, 1) 
        self.res5a_branch2b = self._make_conv_block(512, 512, 3, 1, 2, 2, 4)

        self.out3a = self._make_conv_block(128, 64, 3, 1, 1, 1, 2) 
        self.out5a = self._make_conv_block(512, 64, 3, 1, 4, 4, 2) 
        
        self.ctx_conv1 = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv2 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv3 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv4 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv_final = self._make_conv_block(64, self.num_class, 3, 1, 2, 2, 1)
        self.poolind2d = nn.MaxPool2d(2,2)
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def _make_conv_block(self, in_channel, out_channel, kernel_size, stride, padding, dilation = 1, group = 1):
        layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation ,group),
                               nn.BatchNorm2d(out_channel),
                            nn.ReLU()
        )
        return layers

    # def Bilinear_Interpolation(self, x):
    #     print(x.size())
    #     x = F.interpolate(input, (int(x.size()[2]*2), int(x.size()[3]*2)), mode = 'bilinear')
    #     return x

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x) #256
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) #128
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) #64
        x = self.res3a_branch2b(x)
        out3a_ = self.out3a(x)
        x = self.poolind2d(x)
        x = self.res4a_branch2a(x) #32
        x = self.res4a_branch2b(x)
        # x = self.poolind2d(x)

        x = self.res5a_branch2a(x) #16
        x = self.res5a_branch2b(x)
        x = self.out5a(x)
        # out5a_up2 = self.Bilinear_Interpolation(x)
        out5a_up2 = self.bilinear2d(x)
        # print(out3a_.size())
        # print(out5a_up2.size())
        # out5a_combined = out5a_up2 + out3a_
        out5a_combined = out5a_up2 

        x = self.ctx_conv1(out5a_combined)
        x = self.ctx_conv2(x)
        x = self.ctx_conv3(x)
        x = self.ctx_conv4(x)
        x = self.ctx_conv_final(x)
        x = self.bilinear2d(x)
        x = self.bilinear2d(x)
        x = self.bilinear2d(x)
        x = self.sigmoid(x)
        if (x.size()[2], x.size()[3]) != self.input_size:
            # print((x.size()[2], x.size()[3]),self.img_size)
            x = F.interpolate(x, self.input_size, mode='nearest')
        # print(x.size())

        return x
