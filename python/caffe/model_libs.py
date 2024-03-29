import os
#import config
#import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

kwargs = {
    'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
    'weight_filler': dict(type='xavier'),
    'bias_filler': dict(type='constant', value=0)
}
def get_fc1(net, last_layer, num_classes, fc_type, input_channel=512):
    body = last_layer
    if fc_type== "E":
        body = L.BatchNorm(body, name="bn1")

        body = L.Dropout(body, dropout_ratio=0.4)
  #      net.drop_ouput = net.bn1
        fc1 = L.InnerProduct(body, num_output=num_classes, name="pre_fc1",**kwargs)
        fc1 = net.fc1 = L.BatchNorm(fc1, name="fc1")

    return fc1





def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

kwargs = {
    'param': [dict(lr_mult=1, decay_mult=1)],
    'weight_filler': dict(type='xavier')}

def BoyNetBody(net, from_layer, num_classes):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)}
    #assert from_layer in net.keys()  # 112 x 112
    net.conv1_1 = L.Convolution(from_layer, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    net.conv1_3 = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_3 = L.ReLU(net.conv1_3, in_place=True)
    net.conv1_4 = L.Convolution(net.relu1_3, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_4 = L.ReLU(net.conv1_4, in_place=True)

    name = 'pool1'
    net.pool1 = L.Pooling(net.relu1_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    print num_classes
    # 56 x 56
    net.conv2_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    # 28 x 28
    net.conv2_3 = L.Convolution(net.relu2_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu2_3 = L.ReLU(net.conv2_3, in_place=True)
    net.conv2_4 = L.Convolution(net.relu2_3, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu2_4 = L.ReLU(net.conv2_4, in_place=True)

    name = 'pool2'
    net[name] = L.Pooling(net.relu2_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    print num_classes
    # 14 x 14
    net.conv3_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)

    name = 'pool3'
    net[name] = L.Pooling(net.relu3_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    dilation = 1

    # 7 x 7

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    name = 'pool5'
    net[name] = L.Pooling(net.relu4_2, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)

    # 3 x 3
    dilation = dilation * 6
    kernel_size = 3
    num_output = 512
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.fc6 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    name = "output_feature"

    net[name] = L.InnerProduct(net.relu6,num_output=num_classes,weight_filler=dict(type='xavier'))
    return net[name]

def Conv(net, from_layer, num_output=1, kernel_size=1, stride=1, pad=0, num_group=1, name=None, suffix=""):
    type= "Convolution"
    if num_group != 1:
        type = "DepthwiseConvolution"
    conv = L.Convolution(from_layer, type= type, num_output=num_output, kernel_size=kernel_size, group=num_group, stride=stride, pad=pad, bias_term=False, name="%s%s_conv2d"%(name, suffix) , engine=P.Convolution.Engine.CAFFE, **kwargs)
    bn = L.BatchNorm(conv, name="%s%s_batchnorm"%(name, suffix))
    act = L.ReLU(bn, name="%s%s_relu"%(name, suffix))
    return act

def Linear(net, from_layer, num_output=1, kernel_size=1, stride=1, pad=0, num_group=1, name=None, suffix=""):
    type= "Convolution"
    if num_group != 1:
        type = "DepthwiseConvolution"
    conv = L.Convolution(from_layer, num_output=num_output, kernel_size=kernel_size, group=num_group, stride=stride, pad=pad, bias_term=False, name="%s%s_conv2d"%(name,suffix), type=type,engine=P.Convolution.Engine.CAFFE, **kwargs)
    bn = L.BatchNorm(conv, name="%s%s_batchnormal"%(name,suffix))
    return bn

def DResidual(net, from_layer, num_output=1, kernel_size=3, stride=2, pad=1, num_group=1, name=None, suffix=""):
    conv = Conv(net, from_layer, num_output=num_group, kernel_size=1, pad=0, stride=1, name="%s%s_conv_sep"%(name, suffix))
    conv_dw = Conv(net, conv, num_output=num_group, num_group=num_group, kernel_size=kernel_size, pad=pad, stride=stride, name="%s%s_dw"%(name,suffix))
    proj = Linear(net, conv_dw, num_output=num_output, kernel_size=1, pad=0, stride=1, name="%s%s_conv_proj"%(name,suffix))
    return proj

def Residual(net, from_layer, num_block=1, num_output=1, kernel_size=3, stride=1, pad=1, num_group=1, name=None, suffix=""):
    identity = from_layer
    for i in range(num_block):
        shortcut = identity
        conv = DResidual(net, identity, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, num_group=num_group, name="%s%s_block"%(name,suffix), suffix="%d"%i)
        identity = L.Eltwise(conv, shortcut)
    return identity

def FmobileFaceNetBody(net, from_layer, config):
    num_classes = config.emb_size
    fc_type = config.net_output
    blocks = config.net_blocks
    # 112 x 112
    output_layer = Conv(net, from_layer, num_output=64, kernel_size=3, pad=1, stride=2, name="conv_1")
    # 56 x 56
    if blocks[0]==1:
        output_layer = Conv(net, num_group=64, num_output=64, kernel_size=3, pad=1, stride=1, name="conv_2_dw")
    else:
        output_layer = Residual(net, output_layer, num_block=blocks[0], num_output=64, kernel_size=3, stride=1, pad=1, num_group=64, name="res_2")
    output_layer = DResidual(net, output_layer, num_output=64, kernel_size=3, stride=2, pad=1, num_group=128, name="dconv_23")

    # 28 x 28
    output_layer = Residual(net, output_layer, num_block=blocks[1], num_output=64, kernel_size=3, stride=1,pad=1, num_group=128, name="res_3")
    output_layer = DResidual(net, output_layer, num_output=128, kernel_size=3, stride=2, pad=1, num_group=256, name="dconv_34")
    output_layer = Residual(net, output_layer, num_block=blocks[2], num_output=128, kernel_size=3, stride=1, pad=1, num_group=256, name="res_4")
    output_layer = DResidual(net, output_layer, num_output=128, kernel_size=3, stride=2, pad=1, num_group=512, name="dconv_45")
    output_layer = Residual(net, output_layer, num_block=blocks[3], num_output=128, kernel_size=3, stride=1, pad=1, num_group=256, name="res_5")
    output_layer = Conv(net, output_layer, num_output=512, kernel_size=1, pad=0, stride=1, name="conv_6sep")
    fc1 = get_fc1(net, output_layer, num_classes, fc_type)
    return fc1

