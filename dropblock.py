import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask
    

class BasicBlockDrop(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False, enable_sap=False, enable_conv=True):
        super(BasicBlockDrop, self).__init__()
        self.enable_sap = enable_sap
        self.enable_conv = enable_conv
        print("Enable conv in block:", self.enable_conv)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        if enable_sap:
            # self.transform = torch.zeros(planes, planes, 1, 1)
            # for dim in range(planes):
            #     self.transform[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
            # self.transform = nn.Parameter(self.transform)
            self.scales = [torch.ones(planes, inplanes, 1, 1)] + [torch.ones(planes, planes, 1, 1) for _ in range(2)]
            self.scales = nn.ParameterList([nn.Parameter(x) for x in self.scales])

            #self.transSIMPLEs = nn.ParameterList([ nn.Parameter(torch.ones(planes)) for _ in range(3)])

            # bias transforms
            self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(3)])
            self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(planes)) for _ in range(3)])

            if self.enable_conv:
                self.trans1x1s = nn.Parameter(torch.zeros(planes, planes, 1, 1))
                self.trans3x3s = nn.Parameter(torch.zeros(planes, planes, 3, 3))
                self.U = nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)]))
                self.V = nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)]))
                self.S = nn.Parameter( torch.ones(planes, planes, 1) )



                # alfas
                self.alfasCONV = nn.Parameter(torch.zeros(4))
            self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(2)) for _ in range(3)])
            self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(3)])





        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)
    
    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        if self.enable_sap:
            mtl_alfas = F.softmax(self.alfasWEIGHT[0],dim=0)
            bias_alfas = F.softmax(self.alfasBIAS[0], dim=0)
            cweight1 = mtl_alfas[0]*self.conv1.weight + mtl_alfas[1]*self.conv1.weight*self.scales[0] #+ mtl_alfas[2]*self.conv1.weight*self.transSIMPLEs[0].view(self.conv1.weight.shape[:1]+(1,1,1))
            
            default_bias = torch.zeros(self.conv1.weight.size(0), device=x.device)
            cbias1 = bias_alfas[0]*default_bias + bias_alfas[1]*self.transCONST[0] + bias_alfas[2]*self.shifts[0]
            
            out = F.conv2d(x, weight=cweight1, bias=cbias1, stride=self.conv1.stride, padding=self.conv1.padding, 
                            dilation=self.conv1.dilation, groups=self.conv1.groups)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        if self.enable_sap:
            mtl_alfas = F.softmax(self.alfasWEIGHT[1],dim=0)
            bias_alfas = F.softmax(self.alfasBIAS[1], dim=0)
            cweight2 = mtl_alfas[0]*self.conv2.weight + mtl_alfas[1]*self.conv2.weight*self.scales[1] #+ mtl_alfas[2]*self.conv2.weight*self.transSIMPLEs[1].view(self.conv2.weight.shape[:1]+(1,1,1))
            
            default_bias = torch.zeros(self.conv2.weight.size(0), device=x.device)
            cbias2 = bias_alfas[0]*default_bias + bias_alfas[1]*self.transCONST[1] + bias_alfas[2]*self.shifts[1]
            
            out = F.conv2d(out, weight=cweight2, bias=cbias2, stride=self.conv2.stride, padding=self.conv2.padding, 
                            dilation=self.conv2.dilation, groups=self.conv2.groups)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)
        # version
        if self.enable_sap:
            mtl_alfas = F.softmax(self.alfasWEIGHT[2],dim=0)
            bias_alfas = F.softmax(self.alfasBIAS[2], dim=0)
            cweight3 = mtl_alfas[0]*self.conv3.weight + mtl_alfas[1]*self.conv3.weight*self.scales[2] #+ mtl_alfas[2]*self.conv3.weight*self.transSIMPLEs[2].view(self.conv3.weight.shape[:1]+(1,1,1))

            default_bias = torch.zeros(self.conv3.weight.size(0), device=x.device)   
            if not self.enable_conv:
                cbias3 = bias_alfas[0]*default_bias + bias_alfas[1]*self.transCONST[2] + bias_alfas[2]*self.shifts[2]
            else:
                cbias3 = None

            out = F.conv2d(out, weight=cweight3, bias=cbias3, stride=self.conv3.stride, padding=self.conv3.padding, 
                            dilation=self.conv3.dilation, groups=self.conv3.groups)
        else:
            out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)

        if self.enable_sap and self.enable_conv:
            ## TRANSFORM
            # dont forget to use cbias3
            conv_alfas = F.softmax(self.alfasCONV, dim=0)
            conv_svd = torch.matmul(torch.matmul(self.U, torch.diag_embed(self.S)), self.V.transpose(-2, -1))

            # conv transformations 
            out_pad = F.pad(out, (0,2,0,2), mode='constant')
            # Transform input
            out1 = F.conv2d(out, weight=self.trans1x1s, bias=None) # no padding required cuz k=1
            out2 = F.conv2d(out_pad, weight=self.trans3x3s, bias=None)
            out3 = F.conv2d(out_pad, weight=conv_svd, bias=None)

            out = conv_alfas[0]*out + conv_alfas[1]*out1 + conv_alfas[2]*out2 + conv_alfas[3]*out3
            out = bias_alfas[0]*out + bias_alfas[1]*(out+self.transCONST[2]) + bias_alfas[2]*(out+self.shifts[2].unsqueeze(1).unsqueeze(2).repeat(1,out.size(-2),out.size(-1)))
            ###############################

        out = self.maxpool(out)

        if self.drop_rate > 0 and str(x.device) != "cpu":
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    def get_alfas(self):
        if self.enable_sap:
            #self.alfasCONV,
            for b in [self.alfasBIAS, self.alfasWEIGHT]:
                for p in b:
                    yield p
            if self.enable_conv:
                yield self.alfasCONV
        return
        yield

    def transform_params(self):
        if self.enable_sap:
            for p in self.scales:
                yield p
            for p in self.shifts:
                yield p
            #for p in self.transSIMPLEs:
            #    yield p
            for p in self.transCONST:
                yield p
            
            if self.enable_conv:
                yield self.trans1x1s
                yield self.trans3x3s
                yield self.U
                yield self.V
                yield self.S
            
        return
        yield

    def base_params(self):
        return
        yield


class ResNetDrop(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), block=BasicBlockDrop, n_blocks=[1,1,1,1], keep_prob=1.0, avg_pool=True, drop_rate=0.1,
                 dropblock_size=5, use_se=False, nearest_neighbor=False, use_logits=False, normalize=True, adapt=False, simple_linear=False, enable_sap=False, 
                 enable_conv=True, **kwargs):
        super(ResNetDrop, self).__init__()

        self.inplanes = 3
        self.use_se = use_se
        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes
        self.nn = nearest_neighbor
        self.use_logits = use_logits
        self.normalize = normalize
        self.adapt = adapt
        self.simple_linear = simple_linear
        self.enable_sap = enable_sap
        self.enable_conv = enable_conv
        print("Enable conv:", self.enable_conv)
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.adapt:
            if self.use_logits:
                self.adaptable_linear = nn.Linear(64, self.num_classes)
            else:
                self.adaptable_linear = nn.Linear(640, self.num_classes)
            
            if enable_sap:
                self.fintransform = nn.Linear(self.num_classes, self.num_classes)
                self.fintransform.weight.data = torch.eye(self.num_classes)
                self.fintransform.bias.data = torch.zeros(*list(self.fintransform.bias.size()))
                self.fin_alfa = nn.Parameter(torch.zeros(1))

                self.logittransform = nn.Linear(64,64)
                self.logittransform.weight.data = torch.eye(64)
                self.logittransform.bias.data = torch.zeros(*list(self.logittransform.bias.size()))
                self.logit_alfa = nn.Parameter(torch.zeros(1))

                # self.flattransform = nn.Linear(640,640)
                # self.flattransform.weight.data = torch.eye(640)
                # self.flattransform.bias.data = torch.zeros(*list(self.flattransform.bias.size()))

        if enable_sap:
            indim=outdim=3
            if self.enable_conv:
                # Input transforms
                # 1x1 conv
                conv1x1 = torch.zeros(outdim, indim, 1, 1)
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
                self.conv_alfas = nn.Parameter( torch.zeros(4) )
            
                self.bias_const = nn.Parameter(torch.zeros(1).squeeze())
                self.bias_vect = nn.Parameter(torch.zeros(outdim)) 
                self.bias_alfas = nn.Parameter( torch.zeros(3) )

        rnd_input = torch.rand(1,3,84,84)
        rnd_output = self._forward(rnd_input)
        self.in_features = rnd_output.size(1)
        print("NUM IN FEATURES:", self.in_features)


        self.linear = nn.Linear(640, self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))

        # what they use for logits
        self.classifier = nn.Linear(640, 64)
        self.classifier.bias.data = torch.zeros(*list(self.classifier.bias.size()))


    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se, enable_sap=False, enable_conv=self.enable_conv)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se, enable_sap=False, enable_conv=self.enable_conv)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se, enable_sap=False, enable_conv=self.enable_conv)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se, enable_sap=False, enable_conv=self.enable_conv)
            layers.append(layer)

        return nn.Sequential(*layers)

    def _forward(self, x):
        if self.enable_sap:
            if self.enable_conv:
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
                x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
            #print(x.size(), bias_const.size(), bias_vect.size())
                bias_alfas = F.softmax(self.bias_alfas, dim=0)
                x = bias_alfas[0]*x + bias_alfas[1]*(x+self.bias_const) + bias_alfas[2]*(x+self.bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
            ###################################################
            #  End input transform
            ###################################################

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def _fc_forward(self, x):
        # if self.enable_sap:
        #    x = self.flattransform(x)

        if self.use_logits:
            x = self.classifier(x)

            if self.enable_sap:
                a = torch.sigmoid(self.logit_alfa)
                x = (1-a)*x + a*self.logittransform(x)

        if self.normalize:
            x = normalize(x)
        
        return x


    def forward(self, x, y=None, xquery=None, yquery=None, is_feat=False, return_supp=False):
        x = self._forward(x)

        if return_supp:
            if self.use_logits:
                x = self.classifier(x)

            if self.normalize:
                x = normalize(x)
            
            return x

        if not self.nn or self.simple_linear:
            x = self.linear(x)
            if self.simple_linear:
                return x
            if self.normalize:
                return normalize(x)
            return x
        else:
            xquery = self._forward(xquery)

            if self.use_logits:
                x = self.classifier(x)
                xquery = self.classifier(xquery)
            
                if self.enable_sap:
                    a = torch.sigmoid(self.logit_alfa)
                    x = (1-a)*x + a*self.logittransform(x)
                    xquery = (1-a)*xquery + a*self.logittransform(xquery)

            if self.normalize:
                x = normalize(x)
                xquery = normalize(xquery)

            if self.adapt:
                out = self.adaptable_linear(xquery)
                if self.enable_sap:
                    a = torch.sigmoid(self.fin_alfa)
                    out = (1-a)*out + a*self.fintransform(out)
                return out

            preds = []
            for c in range(self.num_classes):
                indices = y == c
                pred = torch.max(-torch.cdist(xquery, x[indices]),dim=1).values.unsqueeze(1)
                preds.append(pred)
            preds = torch.cat(preds, dim=1)
            return preds
    
    def base_params(self):
        return
        yield

    def transform_params(self):
        # yield base params from all layers
        # for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #     for el in l:
        #         for param in el.transform_params():
        #             yield param

        # yield base params from linear (these are not literally transform params but it allows the model
        # to update them during task-specific training)
        # for param in self.linear.parameters():
        #     yield param

        if self.enable_sap:
            for p in self.fintransform.parameters():
                yield p
            
            for p in self.logittransform.parameters():
                yield p

            if self.enable_conv:
                for param in [self.conv3x3, self.conv1x1, self.U, self.S, self.V, self.bias_const, self.bias_vect]:
                    yield param
            # else:
            #     for param in [self.bias_const, self.bias_vect]:
            #         yield param

            for group in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in group:
                    for p in block.transform_params():
                        yield p
        else:
            return
            yield
        


    def get_alfas(self):
        if self.enable_sap:
            if self.enable_conv:
                for a in [self.conv_alfas, self.bias_alfas]:
                    yield a
            # else:
            #     for a in [self.bias_alfas]:
            #         yield a

            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for b in layer:
                    for a in b.get_alfas():
                        yield a
            
            yield self.fin_alfa
            yield self.logit_alfa
        return
        yield
