import torch 
model = torch.load('results/saved_models/ERVSR.pytorch')
dic = {}
for k in model:
    dic[k.replace('module.Network.', 'Network.')] = model[k]
torch.save(dic, 'results/saved_models/final.pth')