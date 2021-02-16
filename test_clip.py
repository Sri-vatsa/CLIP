import torch
import clip

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
net = model.visual
#print(net)
input = torch.rand([1, 3, 224, 224], dtype=torch.float32)
input = input.to(device)
output, attn_weights = net(input)

print(output.shape)
print(attn_weights.shape)