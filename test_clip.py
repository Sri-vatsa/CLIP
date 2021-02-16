import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
net = model.visual
print(net)
#input = torch.rand([1, 3, 224, 224], dtype=torch.float16)
#input = input.to(device)
#output = net(input)