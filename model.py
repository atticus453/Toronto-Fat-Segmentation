import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class dsv_module(nn.Module):
    def __init__(self, in_ch, out_ch, in_factor=2):
        super(dsv_module, self).__init__()
        self.conv1 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1,bias=True)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, g, x):
        g1 = self.conv1(g)
        x1 = self.conv2(x)
        out = g1+x1
        return out

    
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, init_f=32):
        super(UNet, self).__init__()

        #n1 = 32 
        n1 = init_f
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')

        self.Conv1 = conv_block(img_ch, filters[0]).to('cuda:0')
        self.Conv2 = conv_block(filters[0], filters[1]).to('cuda:0')
        self.Conv3 = conv_block(filters[1], filters[2]).to('cuda:0')
        self.Conv4 = conv_block(filters[2], filters[3]).to('cuda:0')
        self.Conv5 = conv_block(filters[3], filters[4]).to('cuda:0')

        self.Up5 = up_conv(filters[4], filters[3]).to('cuda:0')
        self.Up_conv5 = conv_block(filters[4], filters[3]).to('cuda:0')

        self.Up4 = up_conv(filters[3], filters[2]).to('cuda:0')
        self.Up_conv4 = conv_block(filters[3], filters[2]).to('cuda:0')

        self.Up3 = up_conv(filters[2], filters[1]).to('cuda:0')
        self.Up_conv3 = conv_block(filters[2], filters[1]).to('cuda:0')

        self.Up2 = up_conv(filters[1], filters[0]).to('cuda:0')
        self.Up_conv2 = conv_block(filters[1], filters[0]).to('cuda:1')

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0).to('cuda:1')


    def forward(self, x):
        # encoding path
        x = x.to('cuda:0')
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # decoding path
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = d2.to('cuda:1')
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out




class AG_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, init_f=32):
        super(AG_UNet, self).__init__()

        #n1 = 32
        n1 = init_f
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')

        self.Conv1 = conv_block(img_ch, filters[0]).to('cuda:0')
        self.Conv2 = conv_block(filters[0], filters[1]).to('cuda:0')
        self.Conv3 = conv_block(filters[1], filters[2]).to('cuda:0')
        self.Conv4 = conv_block(filters[2], filters[3]).to('cuda:0')
        self.Conv5 = conv_block(filters[3], filters[4]).to('cuda:0')

        self.Up5 = up_conv(filters[4], filters[3]).to('cuda:0')
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2]).to('cuda:0')
        self.Up_conv5 = conv_block(filters[4], filters[3]).to('cuda:0')

        self.Up4 = up_conv(filters[3], filters[2]).to('cuda:0')
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1]).to('cuda:0')
        self.Up_conv4 = conv_block(filters[3], filters[2]).to('cuda:0')

        self.Up3 = up_conv(filters[2], filters[1]).to('cuda:0')
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0]).to('cuda:0')
        self.Up_conv3 = conv_block(filters[2], filters[1]).to('cuda:0')

        self.Up2 = up_conv(filters[1], filters[0]).to('cuda:1')
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32).to('cuda:1')
        self.Up_conv2 = conv_block(filters[1], filters[0]).to('cuda:1')

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0).to('cuda:1')


    def forward(self, x):
        x = x.to('cuda:0')
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d3 = d3.to('cuda:1')
        e1 = e1.to('cuda:1')
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out



class DSV_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, init_f=32):
        super(DSV_UNet, self).__init__()

        #n1 = 32
        n1 = init_f
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        #filters = [64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')

        self.Conv1 = conv_block(img_ch, filters[0]).to('cuda:0')
        self.Conv2 = conv_block(filters[0], filters[1]).to('cuda:0')
        self.Conv3 = conv_block(filters[1], filters[2]).to('cuda:0')
        self.Conv4 = conv_block(filters[2], filters[3]).to('cuda:0')
        self.Conv5 = conv_block(filters[3], filters[4]).to('cuda:0')


        self.Up5 = up_conv(filters[4], filters[3]).to('cuda:0')
        self.Up_conv5 = conv_block(filters[4], filters[3]).to('cuda:0')

        self.Up4 = up_conv(filters[3], filters[2]).to('cuda:0')
        self.Up_conv4 = conv_block(filters[3], filters[2]).to('cuda:0')

        self.Up3 = up_conv(filters[2], filters[1]).to('cuda:0')
        self.Up_conv3 = conv_block(filters[2], filters[1]).to('cuda:0')

        self.Up2 = up_conv(filters[1], filters[0]).to('cuda:0')
        self.Up_conv2 = conv_block(filters[1], filters[0]).to('cuda:1')

        self.Dsv_Conv5 = nn.Conv3d(filters[3], output_ch, kernel_size=1, stride=1, padding=0).to('cuda:1')
        self.Dsv_Conv4 = dsv_module(filters[2], output_ch).to('cuda:1')
        self.Dsv_Conv3 = dsv_module(filters[1], output_ch).to('cuda:1')
        self.Dsv_Conv2 = dsv_module(filters[0], output_ch).to('cuda:1')



    def forward(self, x):
        x = x.to('cuda:0')
        
        # endocder
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)


        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)

        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = d2.to('cuda:1')
        d2 = self.Up_conv2(d2)
        
        # sum again
        d5 = d5.to('cuda:1')
        d4 = d4.to('cuda:1')
        d3 = d3.to('cuda:1')
        dsv4 = self.Dsv_Conv5(d5)
        dsv3 = self.Dsv_Conv4(dsv4,d4)
        dsv2 = self.Dsv_Conv3(dsv3,d3)

        out = self.Dsv_Conv2(dsv2,d2)

        return out



class AG_DSV_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, init_f=32):
        super(AG_DSV_UNet, self).__init__()

        #n1 = 32
        n1 = init_f
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1*32]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda:0')

        self.Conv1 = conv_block(img_ch, filters[0]).to('cuda:0')
        self.Conv2 = conv_block(filters[0], filters[1]).to('cuda:0')
        self.Conv3 = conv_block(filters[1], filters[2]).to('cuda:0')
        self.Conv4 = conv_block(filters[2], filters[3]).to('cuda:0')
        self.Conv5 = conv_block(filters[3], filters[4]).to('cuda:0')

        self.Up5 = up_conv(filters[4], filters[3]).to('cuda:0')
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2]).to('cuda:0')
        self.Up_conv5 = conv_block(filters[4], filters[3]).to('cuda:0')


        self.Up4 = up_conv(filters[3], filters[2]).to('cuda:0')
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1]).to('cuda:0')
        self.Up_conv4 = conv_block(filters[3], filters[2]).to('cuda:0')


        self.Up3 = up_conv(filters[2], filters[1]).to('cuda:0')
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0]).to('cuda:0')
        self.Up_conv3 = conv_block(filters[2], filters[1]).to('cuda:0')


        self.Up2 = up_conv(filters[1], filters[0]).to('cuda:1')
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32).to('cuda:1')
        self.Up_conv2 = conv_block(filters[1], filters[0]).to('cuda:1')


        self.Dsv_Conv5 = nn.Conv3d(filters[3], output_ch, kernel_size=1, stride=1, padding=0).to('cuda:0')
        self.Dsv_Conv4 = dsv_module(filters[2], output_ch).to('cuda:0')
        self.Dsv_Conv3 = dsv_module(filters[1], output_ch).to('cuda:0')
        self.Dsv_Conv2 = dsv_module(filters[0], output_ch).to('cuda:0')



    def forward(self, x):
        #h,w,z = 256,256,64
        z, h, w = x.size(2), x.size(3), x.size(4)

        x = x.to('cuda:0')
        # encoding path
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # decoding path
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d3 = d3.to('cuda:1')
        d2 = self.Up2(d3)
        e1 = e1.to('cuda:1')
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # concat again for dsv
        d5 = d5.to('cuda:0')
        d4 = d4.to('cuda:0')
        d3 = d3.to('cuda:0')
        d2 = d2.to('cuda:0')
       
        dsv4 = self.Dsv_Conv5(d5)
        dsv3 = self.Dsv_Conv4(dsv4,d4)
        dsv2 = self.Dsv_Conv3(dsv3,d3)
        out = self.Dsv_Conv2(dsv2,d2)

        return out







