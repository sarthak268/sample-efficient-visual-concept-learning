import torch
import torch.nn as nn
import torchvision.models.vgg as models

class VGG_Net(torch.nn.Module):
    def __init__(self):
        super(VGG_Net, self).__init__()

        self.create_vgg_net()

    def create_vgg_net(self):
        vgg16 = models.vgg16(pretrained=True)
                
        self.vgg_network_feature_extractor = vgg16.features
        self.vgg_network_avg_pool = vgg16.avgpool
        self.vgg_network_classifier = vgg16.classifier[:-2]

    def forward(self, x):
        x = self.vgg_network_feature_extractor(x)
        x = self.vgg_network_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.vgg_network_classifier(x)
        return x

def classifierWeightsInit(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.001)
        layer.bias.data.fill_(-6.58)

def singleClassifierWeightsInit(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.001)
        layer.bias.data.fill_(-6.58)

class Classifier(torch.nn.Module):
    def __init__(self, opt, num_nodes_removed=0, use_dropout=True, no_graph_data=False, only_vit=False):
        super(Classifier, self).__init__()

        self.opt = opt
        self.out_size = self.opt.vocab_size - num_nodes_removed
        self.representation_size = 4096 if opt.load_net_type == 'VGG' else 1000
        self.use_dropout = use_dropout
        self.no_graph_data = no_graph_data
        self.only_vit = only_vit
        self.createClassifier()

    def createClassifier(self):
        dropout = nn.Dropout(0.5) if self.use_dropout else nn.Identity() 
        if self.no_graph_data:
            self.fc8 = nn.Linear(self.representation_size + self.opt.detector_size, self.out_size)
        elif self.only_vit:
            self.fc8 = nn.Linear(self.representation_size, self.out_size)
        else:
            graph_size = self.opt.context_dim * self.opt.vocab_size
            self.fc8 = nn.Linear(self.representation_size + self.opt.detector_size + graph_size, self.out_size)

        sigmoid = nn.Sigmoid()
        self.fc_classifier = nn.Sequential(dropout, self.fc8, sigmoid)

    def updateClassifierNovelClass(self, single_class_classifier):
        graph_size = self.opt.context_dim * self.opt.vocab_size
        dropout = nn.Dropout(0.5) if self.use_dropout else nn.Identity() 
        sigmoid = nn.Sigmoid()

        self.out_size += 1
        fc8_new = nn.Linear(self.representation_size + self.opt.detector_size + graph_size, self.out_size)
        
        with torch.no_grad():
            fc8_new.weight[:self.out_size - 1, :] = self.fc_classifier[1].weight
            fc8_new.bias[:-1] = self.fc_classifier[1].bias
            fc8_new.weight[-1, :] = single_class_classifier.fc_classifier[1].weight
            fc8_new.bias[-1] = single_class_classifier.fc_classifier[1].bias
        self.fc8 = fc8_new
        
        self.fc_classifier = nn.Sequential(dropout, self.fc8, sigmoid)
        
        self.fc_classifier = self.fc_classifier.to(self.opt.device)
    
    def forward(self, vgg_input, detect_input=None, graph_input=None):
        if self.no_graph_data:
            fc7_plus = torch.cat((vgg_input, detect_input), dim=-1)
        elif self.only_vit:
            fc7_plus = vgg_input
        else:
            fc7_plus = torch.cat((vgg_input, detect_input, graph_input), dim=-1)
        out = self.fc_classifier(fc7_plus)
        return out

class Classifier_Single_Class(torch.nn.Module):
    def __init__(self, opt):
        super(Classifier_Single_Class, self).__init__()

        self.opt = opt
        self.representation_size = 4096 if opt.load_net_type == 'VGG' else 1000
        self.createClassifier()
        
    def createClassifier(self):
        dropout = nn.Dropout(0.5)
        graph_size = self.opt.context_dim * self.opt.vocab_size
        fc8 = nn.Linear(self.representation_size + self.opt.detector_size + graph_size, 1)

        sigmoid = nn.Sigmoid()
        self.fc_classifier = nn.Sequential(dropout, fc8, sigmoid)
    
    def forward(self, vgg_input, detect_input, graph_input):
        fc7_plus = torch.cat((vgg_input, detect_input, graph_input), dim=-1)
        return self.fc_classifier(fc7_plus)


if (__name__ == '__main__'):
    from args.args_graph import opt
    
    vgg_net = VGG_Net(opt)
    classifier = Classifier(opt)
    print (classifier.fc_classifier)