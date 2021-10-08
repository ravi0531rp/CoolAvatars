import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image(path, img_transform , size=(300,300)):
  image = Image.open(path)
  image = image.resize(size, Image.LANCZOS)
  image = img_transform(image).unsqueeze(0)  # 400,400 -> 1,400,400
  return image.to(device)

def get_gram(m): # m => batch size , C , H , W
  b_size , c, h, w = m.size()
  m = m.view(c, h*w)
  m = torch.mm(m, m.t())
  return m

def denorm_image(inp):
  inp = inp.numpy().transpose((1,2,0)) # C, H, W -> H, W, C
  mean = np.array([0.485 , 0.456 , 0.406])
  std = np.array([0.229 , 0.224 , 0.225])
  inp = inp*std + mean   # doing the reverse of (x-mean)/sigma
  inp = inp.clip(inp,0,1)
  return inp

class FeatureExtractor(nn.Module):
  def __init__(self):
    super(FeatureExtractor,self).__init__()
    self.selected_layers = [3,8,15,22]  # all the relu activation layers are chosen
    self.vgg = models.vgg16(pretrained=True).features
  
  def forward(self, x):
    layer_feats = []
    for layer_num,layer in self.vgg._modules.items():
      x = layer(x)
      if int(layer_num) in self.selected_layers:
        layer_feats.append(x)
    return layer_feats



img_transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485 , 0.456 , 0.406) , std=(0.229 , 0.224 , 0.225))
                                    ]
                                   )
content_image = get_image('/content/drive/MyDrive/NeuralStyleTransfer/content.jpg', img_transform)
style_image = get_image('/content/drive/MyDrive/NeuralStyleTransfer/style.jpg', img_transform)

generated_image = content_image.clone()  
generated_image.requires_grad = True

opt = torch.optim.Adam([generated_image] , lr = 0.003 , betas = [0.5, 0.999])
encoder = FeatureExtractor().to(device)

for p in encoder.parameters():
  p.requires_grad = False

style_weight = 100
cont_weight = 1

for epoch in range(5000):
  content_features = encoder(content_image)
  style_features = encoder(style_image)
  generated_features = encoder(generated_image)

  cont_loss = torch.mean((content_features[-1] - generated_features[-1])**2)

  style_loss = 0
  for gf, sf in zip(generated_features,style_features):
    _,c,h,w = gf.size()
    gram_gf = get_gram(gf)
    gram_sf = get_gram(sf)
    style_loss += torch.mean((gram_gf - gram_sf)**2)/(c*h*w)

  loss = cont_weight*cont_loss + style_weight*style_loss
  
  opt.zero_grad()
  loss.backward()
  opt.step()
  if epoch%100 == 0:
    print(f"Epoch is {epoch} , Content Loss is {cont_loss} , Style Loss is {style_loss} , Total Loss is {loss}")
    print("==========================================================================================================")

