# PyTorch imports
import torch
import torch.nn as nn  # For defining neural network layers and models
import torch.nn.functional as F  # For using functional operations like adaptive pooling

# Timm for pre-trained model handling
import timm


import threading  


class eca_nfnet_l0(nn.Module):
    def __init__(self):

        super(eca_nfnet_l0, self).__init__()

        self.model = timm.create_model("hf_hub:timm/eca_nfnet_l0", pretrained=True)
        self.classifier = nn.Linear(self.model.head.fc.in_features, 1, bias=True)


        self.attention=nn.Conv2d(2,1,kernel_size=1, bias=True)
        

        layer_name = 'final_conv'  # Example layer name, adjust based on actual model architecture
        
        self.features = None
        #self.features = {}

        # Adjusted hook registration
        self.model.final_act.register_forward_hook(self.get_features)  
        #self.model.final_conv.register_forward_hook(self.get_features)
        

    def get_features(self, module, input, output):
        self.features=output
        #self.features[threading.get_ident()] = output.detach()

        
    def getAttFeats(self,att_map,features):
        features=0.5*features+0.5*(att_map*features)
        return features


    def forward(self, x):

        
        outputs={}
        
        dummy = self.model(x)

        features= self.features#[threading.get_ident()]
        fg_att = self.attention(torch.cat((torch.mean(features, dim=1).unsqueeze(1), torch.max(features, dim=1)[0].unsqueeze(1)), dim=1))
        fg_att = torch.sigmoid(fg_att)
        features = self.getAttFeats(fg_att, features)
        
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        outputs['logits']=out
        outputs['feat'] =features
        return  out #outputs