print('This is YOLO API')
from darknet import *
import os.path as osp
import os 

class YOLO(object):
    def __init__(self, cfgfile, weightfile = ''):
        self.net = Darknet(cfgfile, weightfile)
        self.device = torch.device("cpu")
    def compare_weights(self, file1, file2):
        import filecmp
        are_same = filecmp.cmp(file1, file2, shallow=False)
        if are_same:
            print('Files are equal')
        else:
            print('Files differ')

    # We need a method for loading images
    # Consider a dataset of 10k images
    # we load e.g. 64 images at once (batch),
    # train on it, load next batch, etc.
    # So, given list of image names, we load the batch and do preprocessing
    def load_batch(images):

        start_time = time.time()
        try:
            imlist = [osp.join(osp.realpath('.'), images, img) \
                        for img in os.listdir(images)]
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()

        loaded_ims = [cv2.imread(x) for x in imlist]

        load_batch_time = time.time()

        return loaded_ims
        

model = YOLO('cfg/yolov3.cfg', 'yolov3.weights')
# CUDA = False
CUDA = torch.cuda.is_available()
print('current cuda device:', torch.cuda.current_device())
if CUDA:
    model.net.cuda()
model.net.save_weights('yolov3_train.weights')
model.compare_weights('yolov3.weights', 'yolov3_train.weights')


# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# print('Importing COCO')
# start = time.time()
# cap = dset.CocoDetection(
#         root = 'scripts/coco/images/val2014/',
#         annFile = 'scripts/coco/annotations/instances_val2014.json',
#         transform=transforms.ToTensor())

# print('Import time', time.time() - start)
# print('Number of samples: ', len(cap))
# img, target = cap[3] # load 4th sample


# print("Image Size: ", img.size())
# print(target)


# data_loader = torch.utils.data.DataLoader(
#                     cap,
#                     batch_size=4,
#                     shuffle=True,
#                     num_workers=4)
