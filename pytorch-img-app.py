import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

## image classifier ##
# followed PyTorch in 100 Seconds by Fireship
# https://www.youtube.com/watch?v=ORMx45xqWkA

# create a constructor "class" and define a new class "HiMom" 
# that "inherits" from the nueral network "nn" "module" class
class HiMom(nn.Module):
  def __init__(self):
    super()._init__()
    # flatten will take a multi-dimesional layer and flatten 
    # to single 1-dimension
    self.flatten = nn.Flatten()
    # sequenctial will create a container of layers that data 
    # will flow through. Each layer has multiple nodes, where 
    # each node is a "mini" statistical model. The model will
    # flow each data point to guess an output.
    self.linear_relu_stack = nn.Sequential(
      # Linear is a fully connnected layer that takes a 
      # flattened 28x28 image and transforms it into an 
      # output of 512
      nn.Linear(28*28, 512),
      # ReLu is a "non-linear activation function" to signify 
      # the above layer is important and should output the 
      # complete node, not just zero "0"
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      # the final layer is a fully connected layer will output
      # the "ten labels" the model is trying to predict   
      nn.Linear(512, 10),
    )
  # define a forward method/function that describes the flow 
  # of data
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
# substantiate the model to a GPU, Nvdia"cuda"
model = HiMom().to("cuda")

X = some_data # PASS IN SOME USER DATA HERE

logits = model(X) # will auto call the forward method/function
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax (1)

print (f"And my prediction is... {y_pred}")