import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as TF
from cv2 import imread, imwrite
from torchvision.utils import draw_bounding_boxes, save_image

device = 'cuda'

CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class detector():

  def __init__(self, model_path,threshold=0.3):
    self.threshold = threshold
    self.model = self.load_model(model_path)
  
  def load_model(self, model_path=None):
    """   
    Loads a pretrained model and state_dict if desired 

    Todo: implement channels for IR image data
    implement channels polarimetric data"""

    # print("Loading model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # print("Loading model...done")

    # get the number of input features 
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # define a new head for the detector with required number of classes
    # num_classes = len(CLASSES)
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    if model_path is not None:
      print("Loading model from:", model_path)
      model.load_state_dict(torch.load(model_path), map_location=device)

    return model

  def train(self, dataloader, optimizer, scheduler, num_epochs):
    # loads a custom dataset and trains the model with it

    "Not implemented"
    return None

  def detect(self, image):
    """
    image: tensor
    """

    transform=T.Compose([
              T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    norm_tensor = transform(image.type(torch.FloatTensor)) #normalize the image
    image_tensor = (norm_tensor - torch.min(norm_tensor))/(torch.max(norm_tensor) - torch.min(norm_tensor))*(1 - 0) + 0 # min max scaling from 0 to 1

    if len(image_tensor.shape) == 3:   #if input is single image, wrap in a batch
      image_tensor = image_tensor.unsqueeze(0)
    #else if input is a batch, do nothing

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
      self.model.to(device)
      self.model.eval()
      outputs = self.model(image_tensor)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs] #move outputs to cpu

    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    # filter out boxes according to threshold
    conf_mask = scores > self.threshold
    # print (scores)
    boxes = boxes[conf_mask]
    labels = labels[conf_mask]

    # get all the predicited class names
    pred_classes = [CLASSES[i] for i in labels.cpu()]

    return boxes, pred_classes

if (__name__ == '__main__'):
  # img_path = "../pytorch-faster-rcnn/COCODevKit/train2017/000000000009.jpg"
  # img_path = "../pytorch-faster-rcnn/COCODevKit/train2017/000000000025.jpg"
  # img_path = '../novel_cls/577932.jpg'
  img_path = '../class_examples/hat/data_498399.jpg'
  predictor = detector(threshold = .3, model_path = None)
  img_tensor = torch.from_numpy(imread(img_path)).permute(2, 0, 1)
  boxes, pred_cls = predictor.detect(img_tensor)
  print (pred_cls)
