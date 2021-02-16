import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
net = model.visual

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('')
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    try:
      print('input size:', input[0].size())
    except:
      print('input size:', input[0][0].size(), ' ', input[0][1].size())
    try:
      print('output size:', output.data.size())
    except:
      print('output size:', output[0].data.size(), ' ', output[1].data.size())

for module in net.modules():
  module.register_forward_hook(printnorm)

#print(net)
input = torch.rand([1, 3, 224, 224], dtype=torch.float32)
input = input.to(device)
output, attn_weights = net(input)

#print(output.shape)
#print(attn_weights.shape)