
import numpy as np


import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd #build a computational graph
import torch.optim as optim # optimization package
import time
from util import *

def cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = autograd.Variable(img_)                     # Convert to Variable
    return img_

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self, cfgfile, weightsfile = ''):
        print('Darknet(self,', cfgfile, ', ', weightsfile, ')')

        start_time = time.time()
        super(Darknet, self).__init__()
        
        self.blocks = self.extract_dict_list(cfgfile)
        self.create_modules(self.blocks)
        
        cfg_time = time.time()
        
        self.init_weights(weightsfile)
        
        weights_time = time.time()

        print('time for reading cfg', cfg_time - start_time)
        print('time for initializing weights', weights_time - start_time)
    
    
    
    

    def forward(self, x, CUDA): #CUDA is boolean
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0     #This is explained a bit later
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
        
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':        

                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])

                #Get the number of classes
                num_classes = int(module["classes"])

                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections
   
    def extract_dict_list(self, cfgfile):
        file = open(cfgfile, 'r')
        lines = file.read().split('\n')                 # Read in all        
        lines = [x for x in lines if len(x) > 0]        # non-empty
        lines = [x for x in lines if x[0] != '#']       # non-comment lines, and
        lines = [x.rstrip().lstrip() for x in lines]    # remove trailing/heading ws
        block = {}                                      # Define single block as dict
        blocks = []                                     # and blocks list

        for line in lines:
            if line[0] == "[":               
                if len(block) != 0:          
                    blocks.append(block)                # Send finished block to list
                    block = {}                          # Start new block
                block["type"] = line[1:-1].rstrip()     # set type
            else:
                key,value = line.split("=")             # parse attributes
                block[key.rstrip()] = value.lstrip()    # add attributes

        blocks.append(block)                            # Send last block
        return blocks
   
    def create_modules(self, blocks):

        self.net_info = blocks[0]
        self.module_list = nn.ModuleList()
        
        print('net_info', self.net_info)
        

        output_depths = []
        output_widths = []
        output_heights = []

        prev_depth = int(self.net_info['channels'])     # Depth of the prev. layer (input)
        prev_width = int(self.net_info['width'])        # Width of the prev. layer (input)
        prev_height = int(self.net_info['height'])      # Height of the prev. layer (input)
        

        for index, x in enumerate(blocks[1:]):          # For each block
            module = nn.Sequential()                    # Make a module that may contain
                                                        # multiple layers (conv, relu, etc.)
            if (x['type'] == 'convolutional'):          # Parse conv layer:
                activation = x['activation']            # set activation (e.g. leaky)
                try:
                    batch_normalize = \
                        int(x['batch_normalize'])       # try setting batch_normalize (0 or 1)
                    bias = False                        # if set, bias is set by normalization
                except:
                    batch_normalize = 0                 # no normalization?
                    bias = True                         # then need own bias
                cur_depth = int(x['filters'])           # 
                padding = int(x['pad'])                 # 1 for same, 0 for none
                stride = int(x['stride'])       
                kernel_size = int(x['size'])            # F = filter size

                if padding:                             # padding either same or s.t. width /= 2    
                    pad = (kernel_size - 1) // 2        # if stride = 2, W' = (W - F + 2P)/S + 1
                                                        # = (W - F + F -1)/S + 1 = W/2 + 1/2
                                                        # if s = 1, W' = W - 1 + 1 = W
                else:
                    pad = 0

                cur_width = prev_width - kernel_size + 2*pad + stride
                cur_width //= stride
                cur_height = prev_height - kernel_size + 2*pad + stride
                cur_height //= stride

                conv = nn.Conv2d(prev_depth, cur_depth,     # Make conv module: output different shape
                    kernel_size, stride, pad, bias=bias)

                module.add_module('conv_{0}'.format(index), conv)
                if batch_normalize:
                    bn = nn.BatchNorm2d(cur_depth)
                    module.add_module('batch_norm_{0}'.format(index), bn)   # output same shape
                if activation == 'leaky':
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module('leaky_{0}'.format(index), activn)    # output same shape

            elif (x["type"] == "upsample"):             # Parse Upsample layer
                stride = int(x["stride"])
                upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
                module.add_module("upsample_{}".format(index), upsample)
                cur_width = (prev_width * 2) // 1
                cur_depth = prev_depth
                cur_height = (prev_height * 2) // 1

            #If it is a route layer
            elif (x["type"] == "route"):
                x["layers"] = x["layers"].split(',')
                #Start  of a route
                start = int(x["layers"][0])
                #end, if there exists one.
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0

                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)

                #Positive anotation
                if start > 0: 
                    start = start - index
                if end > 0:
                    end = end - index      
                if end < 0:
                    cur_depth = output_depths[index + start] + output_depths[index + end]
                else:
                    cur_depth = output_depths[index + start]
                
                cur_width = output_widths[index + start]
                cur_height = output_heights[index + start]

            #shortcut corresponds to skip connection
            elif x["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)
                cur_depth = prev_depth
                cur_height = prev_height
                cur_width = prev_width
              #Yolo is the detection layer
            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)
                
            self.module_list.append(module)
            
            prev_width = cur_width
            prev_height = cur_height
            prev_depth = cur_depth
            output_heights.append(cur_height)
            output_widths.append(cur_width)
            output_depths.append(cur_depth)
      

    def init_weights(self, weightsfile = ''):
        print('init_weights(self,', weightsfile, ')')

        if len(weightsfile) > 0:
            self.load_weights(weightsfile)
            return
        # If no file given, start with random weights      
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                   
                    # BatchNorm2d params are very good initially
                    # so no need to initialize
                else:
                    conv.bias.data.fill_(0.01)
                nn.init.xavier_uniform(conv.weight.data)

        # No input file: define own header
        header = np.array([0, 2, 0, 0, 0], dtype=np.int32)  # major, minor, subversion, num_images_seen, 0
        self.header = torch.from_numpy(header)
        self.seen = 0
        print('HEADER', header)
   
    def load_weights(self, weightfile):
        print('load_weights(self,', weightfile, ')')
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        print('HEADER', header)
        self.seen = self.header[3]
        
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
        print('ptr', ptr)
   
    def save_weights(self, weightsfile = 'out.weights'):
        print('save_weights(self,', weightsfile, ')')
        start_time = time.time()


        fp = open(weightsfile, 'wb')                                    # write binary

        self.header[3] = self.seen
        self.header.numpy().tofile(fp)                                  # Dump header (int32)
        
        counter = 0                                                     # Count floats (float32)
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":                          # Only conv have weights
                model = self.module_list[i]                             # load layer
                try:
                    batch_normalize = \
                        int(self.blocks[i+1]["batch_normalize"])        # From cfg: batch normalize?
                except:
                    batch_normalize = 0

                conv = model[0]                                         # conv part of layer
                
                if (batch_normalize):
                    bn = model[1]                                       # bn part of layer
                    num_bn_biases = bn.bias.numel()
                    counter += num_bn_biases * 4                        # bn has biases, weights, means, stds
                else:
                    num_biases = conv.bias.numel()                      # biases of conv layer
                    counter += num_biases
                num_weights = conv.weight.numel()                       # weights of conv layer
                counter += num_weights
          
        print('NUMBER OF FLOATS', counter)    
        
        weights = np.ndarray(shape=(1,counter), dtype=np.float32)       # Output array initialization

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
        
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = cpu(bn.bias.data).numpy()
                    weights[0,ptr:ptr + num_bn_biases] = bn_biases.view(dtype=np.float32).reshape(-1, num_bn_biases)
                    ptr += num_bn_biases

                    bn_weights = cpu(bn.weight.data).numpy()
                    weights[0,ptr: ptr + num_bn_biases] = bn_weights.view(dtype=np.float32).reshape(-1, num_bn_biases)
                    ptr  += num_bn_biases

                    bn_running_mean = cpu(bn.running_mean).numpy()
                    weights[0, ptr: ptr + num_bn_biases] = bn_running_mean.view(dtype=np.float32).reshape(-1, num_bn_biases)
                    ptr  += num_bn_biases

                    bn_running_var = cpu(bn.running_var).numpy()
                    weights[0, ptr: ptr + num_bn_biases] = bn_running_var.view(dtype=np.float32).reshape(-1, num_bn_biases)
                    ptr  += num_bn_biases
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = cpu(conv.bias.data).numpy()
                    weights[0, ptr: ptr + num_biases] = conv_biases.view(dtype=np.float32).reshape(-1, num_biases)
                    ptr = ptr + num_biases

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = cpu(conv.weight.data).numpy()
                weights[0, ptr: ptr + num_weights] = conv_weights.view(dtype=np.float32).reshape(-1, num_weights)
                ptr = ptr + num_weights

        print('PTR', ptr)
  
        weights.tofile(fp)
        print('time for saving weights', time.time() - start_time)

