# adapted from https://raw.githubusercontent.com/yinboc/few-shot-meta-baseline/master/models/resnet12.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

        # transformation parameters
        self.trans1x1s = nn.ParameterList([nn.Parameter(torch.zeros(planes, planes, 1, 1)) for _ in range(3)])
        self.trans3x3s = nn.ParameterList([nn.Parameter(torch.zeros(planes, planes, 3, 3)) for _ in range(3)])
        self.transUs = nn.ParameterList([nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)])) for _ in range(3)])
        self.transVs = nn.ParameterList([nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)])) for _ in range(3)])
        self.transS = nn.ParameterList([nn.Parameter( torch.ones(planes, planes, 1) ) for _ in range(3)])

        # element-wise multiplication transformations on the weights of the 3 conv layers
        self.transMTLs = nn.ParameterList([nn.Parameter(torch.ones(planes, inplanes, 1, 1))] + [nn.Parameter(torch.ones(planes, planes, 1, 1)) for _ in range(2)] )
        self.transSIMPLEs = nn.ParameterList([ nn.Parameter(torch.ones(planes)) for _ in range(3)])

        # addition operations
        self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(3)])
        self.transVECT = nn.ParameterList([nn.Parameter(torch.zeros(planes)) for _ in range(3)]) 

        # alfas
        self.alfasCONV = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(3)])
        self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(3)])
        self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(3)])

        # Identity tensor initialization for 1x1, 3x3, SVD
        for b in [self.trans1x1s, self.trans3x3s, self.transUs, self.transVs]:
            for p in b:
                for dim in range(planes):
                    p.data[dim,dim,0,0] = 1


    def get_alfas(self):
        for block in [self.alfasCONV, self.alfasWEIGHT, self.alfasBIAS]:
            for param in block:
                yield param

    def transform_params(self):
        for block in [self.trans1x1s, self.trans3x3s, self.transUs, self.transS, self.transVs, self.transMTLs, self.transSIMPLEs, self.transCONST, self.transVECT]:
            for param in block:
                yield param
    
    def base_params(self):
        for block in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.downsample]:
            for param in block.parameters():
                yield param


    def _forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


    def forward(self, x):
        identity = self.downsample(x)
        for i in range(3):
            # transform weights
            conv1x1, conv3x3, U,S,V, mtl_scale, simple_scale, bias_const, bias_vect = self.trans1x1s[i], self.trans3x3s[i], self.transUs[i], self.transS[i], self.transVs[i], self.transMTLs[i], self.transSIMPLEs[i], self.transCONST[i], self.transVECT[i]
            conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
            simple_scale = simple_scale.view(conv3x3.shape[:1]+(1,1,1))

            # alfas
            conv_alfas, mtl_alfas, bias_alfas = self.alfasCONV[i], self.alfasWEIGHT[i], self.alfasBIAS[i]
            conv_alfas = F.softmax(conv_alfas, dim=0)
            mtl_alfas = F.softmax(mtl_alfas, dim=0)
            bias_alfas = F.softmax(bias_alfas, dim=0)

            # base-weights
            conv, conv_bias = eval(f"self.conv{i+1}.weight"), eval(f"self.conv{i+1}.bias")
            conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

            # compute x with weight transforms
            if i == 0:
                out = F.conv2d(x, weight=conv_weights, bias=conv_bias, padding=eval(f"self.conv{i+1}.padding"), stride=eval(f"self.conv{i+1}.stride"))
            else:
                out = F.conv2d(out, weight=conv_weights, bias=conv_bias, padding=eval(f"self.conv{i+1}.padding"), stride=eval(f"self.conv{i+1}.stride"))
            out = eval(f"self.bn{i+1}(out)")
            if i == 2:
                out = out + identity
            out = self.relu(out)

            # conv transformations 
            out_pad = F.pad(out, (0,2,0,2), mode='constant')
            # Transform input
            out1 = F.conv2d(out, weight=conv1x1, bias=None) # no padding required cuz k=1
            out2 = F.conv2d(out_pad, weight=conv3x3, bias=None)
            out3 = F.conv2d(out_pad, weight=conv_svd, bias=None)

            out = conv_alfas[0]*out + conv_alfas[1]*out1 + conv_alfas[2]*out2 + conv_alfas[3]*out3
            out = bias_alfas[0]*out + bias_alfas[1]*(out+bias_const) + bias_alfas[2]*(out+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,out.size(-2),out.size(-1)))

        out = self.maxpool(out)
        return out


class ResNet12(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), channels=[64, 128, 256, 512], **kwargs):
        super().__init__()

        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes


        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # Input transforms
        indim=outdim=3
        # 1x1 conv
        conv1x1 = torch.zeros(outdim, indim, 1, 1)
        #nn.init.uniform_(conv1x1, a=-1/indim, b=+1/indim)
        for dim in range(outdim):
           conv1x1[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
        self.conv1x1 = nn.Parameter(conv1x1)

        # 3x3 conv
        conv3x3 = torch.zeros(outdim, indim, 3, 3)
        # nn.init.uniform_(conv3x3, a=-1/(indim*9), b=+1/(indim*9))
        for dim in range(outdim):
           conv3x3[dim,dim,0,0] = 1
        self.conv3x3 = nn.Parameter(conv3x3)

        # 3x3 conv SVD
        self.U = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) # shape (outdim, indim, 1, 1)
        self.V = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) 
        for dim in range(indim):
           self.U.data[dim,dim,0,0] = 1
           self.V.data[dim,dim,0,0] = 1

        self.S = nn.Parameter( torch.ones(outdim, indim, 1) )
        self.bias_const = nn.Parameter(torch.zeros(1).squeeze())
        self.bias_vect = nn.Parameter(torch.zeros(outdim)) 
        self.conv_alfas = nn.Parameter( torch.zeros(4) )
        self.bias_alfas = nn.Parameter( torch.zeros(3) )

        rnd_input = torch.rand(1,3,84,84)
        rnd_output = self._forward(rnd_input)

        self.linear = nn.Linear(rnd_output.size(1), self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))

        self.lineartransform = nn.Linear(self.num_classes, self.num_classes)
        self.lineartransform.weight.data = torch.eye(self.num_classes)
        self.lineartransform.bias.data = torch.zeros(*list(self.linear.bias.size()))
        print("Random output size:", rnd_output.size())


    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def base_params(self):
        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in l.base_params():
                yield param

    def transform_params(self):
        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in l.transform_params():
                yield param

        # yield base params from linear
        for param in self.lineartransform.parameters():
            yield param
        
        for param in [self.conv3x3, self.conv1x1, self.U, self.S, self.V, self.bias_const, self.bias_vect]:
            yield param
        
        # yield base params from linear (these are not literally transform params but it allows the model
        # to update them during task-specific training)
        for param in self.linear.parameters():
            yield param

    def get_alfas(self):
        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in l.get_alfas():
                yield param
        
        for a in [self.conv_alfas, self.bias_alfas]:
            yield a

    def _forward(self, x):
        x = self.layer1._forward(x)
        x = self.layer2._forward(x)
        x = self.layer3._forward(x)
        x = self.layer4._forward(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


    def forward(self, x):
        ###################################################
        #  input transform
        ###################################################
        conv_svd = torch.matmul(torch.matmul(self.U, torch.diag_embed(self.S)), self.V.transpose(-2, -1))
        # transform input x 
        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        x1 = F.conv2d(x, weight=self.conv1x1, bias=None) # no padding required cuz k=1
        x2 = F.conv2d(x_pad, weight=self.conv3x3, bias=None)
        x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

        conv_alfas = F.softmax(self.conv_alfas, dim=0)
        bias_alfas = F.softmax(self.bias_alfas, dim=0)
        x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
        #print(x.size(), bias_const.size(), bias_vect.size())
        x = bias_alfas[0]*x + bias_alfas[1]*(x+self.bias_const) + bias_alfas[2]*(x+self.bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
        ###################################################
        #  End input transform
        ###################################################

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        out = self.linear(x)
        out = self.lineartransform(out)
        return out


def resnet12():
    return ResNet12([64, 128, 256, 512])


def resnet12_wide():
    return ResNet12([64, 160, 320, 640])










import torch.nn as nn


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class BlockReal(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out

    def get_alfas(self):
        return
        yield

    def transform_params(self):
        for block in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.downsample]:
            for param in block.parameters():
                yield param
    
    def base_params(self):
        return
        yield

class ResNet12Real(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), channels=[64, 128, 256, 512], **kwargs):
        super().__init__()

        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes


        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        rnd_input = torch.rand(1,3,84,84)
        rnd_output = self._forward(rnd_input)

        self.linear = nn.Linear(rnd_output.size(1), self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))

        print("Random output size:", rnd_output.size())

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = BlockReal(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block


    def _forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = self.linear(x)
        return x


    def base_params(self):
        return
        yield

    def transform_params(self):
        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in l.transform_params():
                yield param

        # yield base params from linear (these are not literally transform params but it allows the model
        # to update them during task-specific training)
        for param in self.linear.parameters():
            yield param

    def get_alfas(self):
        return
        yield

