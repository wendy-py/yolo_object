import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2, os, time
import matplotlib.pyplot as plt

weights_file = 'yolov3.weights'
class_file = 'coco.names'
cfg_file = 'yolov3.cfg'
input_dir = 'input'
output_dir = 'output'
nms_conf = 0.5

def parse_cfg(config_file):
    file = open(config_file, 'r')
    file = file.read().split('\n')
    file = [line for line in file if len(line) > 0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    final_list = []
    element_dict = {}
    for line in file:
        if line[0] == '[':
            if len(element_dict) != 0:
                final_list.append(element_dict)
                element_dict = {}
            element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
        else:
            val = line.split('=')
            element_dict[val[0].rstrip()] = val[1].lstrip()
    final_list.append(element_dict)

    return final_list


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_model(blocks):
    net_info = blocks[0]
    channels = 3
    filters = 0
    output_filters = []
    modulelist = nn.ModuleList()

    for i, block in enumerate(blocks[1:]):
        seq = nn.Sequential()
        if block["type"] == "convolutional":
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            bias = False if ("batch_normalize" in block) else True
            padding = (kernel_size - 1) // 2

            conv = nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias)
            seq.add_module("conv_{0}".format(i), conv)

            if "batch_normalize" in block:
                bn = nn.BatchNorm2d(filters)
                seq.add_module("batchnorm_{0}".format(i), bn)

            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                seq.add_module("leaky_{0}".format(i), act)

        elif block["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            seq.add_module("upsample_{}".format(i), upsample)

        elif block["type"] == 'route':
            block['layers'] = block['layers'].split(',')
            block['layers'][0] = int(block['layers'][0])
            start = block['layers'][0]
            if len(block['layers']) == 1:
                filters = output_filters[i + start]

            elif len(block['layers']) > 1:
                block['layers'][1] = int(block['layers'][1]) - i
                end = block['layers'][1]
                filters = output_filters[i + start] + output_filters[i + end]

            route = DummyLayer()
            seq.add_module("route_{0}".format(i), route)

        elif block["type"] == "shortcut":
            shortcut = DummyLayer()
            seq.add_module("shortcut_{0}".format(i), shortcut)

        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(",")
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            block["anchors"] = anchors

            detect = DetectionLayer(anchors)
            seq.add_module("detect_{0}".format(i), detect)

        modulelist.append(seq)
        output_filters.append(filters)
        channels = filters

    return net_info, modulelist


def prediction(x, inp_dim, anchors, num_classes, cuda=False):
    batch_size = x.size(0)
    grid_size = x.size(2)
    stride = inp_dim // x.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = x.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # the dimension of anchors is wrt original image.
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # add centre
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # width and height
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride
    return prediction


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_model(self.blocks)
        self.header = None
        self.seen = None

    def forward(self, x):
        modules = self.blocks[1:]
        detections = torch.tensor([])
        outputs = {}
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                if len(layers) > 1:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = prediction(x.data, inp_dim, anchors, num_classes)
                detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i - 1]

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. Images seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except KeyError:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# returns Intercept over Union of two bounding boxes
def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # take only values above a particular threshold and set rest everything to zero
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # (center x, center y, height, width) attributes of our boxes,
    # to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    output = torch.tensor([])

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # take only those rows with maximum class probability and corresponding index
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        img_classes = unique(image_pred_[:, -1])

        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort them based on probability
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]  # getting index
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                # Get the IOUs of all boxes that come after
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            out = torch.cat(seq, 1)
            output = torch.cat((output, out))

    return output


# function to load the classes
def load_classes(file):
    fp = open(file, "r")
    names = fp.read().split("\n")[:-1]
    return names


# function to convert an image from opencv format to torch format
def prep_image(img, inp_dim):
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

# function to resize image with unchanged aspect ratio using padding (128,128,128)
def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

CUDA = torch.cuda.is_available()
batch_size = 2

# Set up the neural network
model = Darknet(cfg_file)
model.load_weights(weights_file)
print("model and weights loaded")
classes = load_classes(class_file)
print(f"{len(classes)} classes loaded")
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [os.path.join(os.path.realpath('.'), input_dir, img) for img in os.listdir(input_dir)]
except NotADirectoryError:
    imlist = os.path.join(os.path.realpath('.'), input_dir)
except FileNotFoundError:
    print("No file or directory with the name {}".format(input_dir))
    exit()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
load_batch = time.time()

# preparing images
# [[image,original_image,dim[0],dim[1]]]
batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
im_batches = [x[0] for x in batches]  # list of resized images
orig_ims = [x[1] for x in batches]  # list of original images
im_dim_list = [x[2] for x in batches]  # dimension list
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)  # repeat twice

if CUDA:
    im_dim_list = im_dim_list.cuda()

# converting image to batches
reminder = 0
if len(im_dim_list) % batch_size:
    reminder = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + reminder
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))]))
                  for i in range(num_batches)]

output = torch.tensor([])
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # Transform the predictions as described in the YOLO paper
    # flatten the prediction vector
    # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
    # Put every proposed box as a row.
    with torch.no_grad():
        prediction = model(batch)

    prediction = write_results(prediction, confidence=0.5, num_classes=80, nms_conf=nms_conf)

    if type(prediction) == int:
        continue

    prediction[:, 0] += i * batch_size
    output = torch.cat((output, prediction))

    if CUDA:
        torch.cuda.synchronize()


# Before we draw the bounding boxes, the predictions contained in our output tensor
# are predictions on the padded image, and not the original image. Merely, re-scaling them
# to the dimensions of the input image won't work here. We first need to transform the
# co-ordinates of the boxes to be measured with respect to boundaries of the area on the
# padded image that contains the original image

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0, 0, 255)
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 0, 0], 1)
    return img


list(map(lambda x: write(x, orig_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(output_dir, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, orig_ims))

torch.cuda.empty_cache()

img1 = cv2.imread('output/det_dog-cycle-car.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.imshow(img1)
plt.show()

img2 = cv2.imread('output/det_sky.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.imshow(img2)
plt.show()
