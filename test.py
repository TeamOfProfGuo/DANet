import torchvision.models as models
vgg16 = models.vgg16()
resnet18 = models.resnet18()


def dl(data='l1', **kwargs):
    print(kwargs)
    print(data)

    def dl1(k1='0', **kwargs):
        print(k1)
        print(kwargs)
    dl1(**kwargs)


kw = {'data': 'voc', 'k1' :'k_1', 'k2':'check'}
dl(**kw)

def tp():
    pass


def main(name='dl', **kwargs):
    dt ={
        'dl': dl,
        'tp': tp
    }
    dt[name](**kwargs)

kw = {'data': 'voc', 'aux' :False}

main(**kw)



def test(*args):
    a, b = tuple(args)
    print('first {}'.format(a))
    print('2nd {}'.format(b))

a = ['hi', 'hi2']
test(['hi', 'hi2'])



t = torch.tensor([[1,2],[3,4]])
torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))

import os
path= os.path.expanduser('~/Documents/semantic_seg/models/DANet/encoding/models/resnet50-19c8e357.pth')

model.load_state_dict(torch.load(path), strict=False)
stored = torch.load(path)
params = model.state_dict()

kwargs = {'os':8, 'norm_layer':SyncBatchNorm}
pretrain = get_backbone(backbone, pretrained=True, dilated = True, **kwargs)