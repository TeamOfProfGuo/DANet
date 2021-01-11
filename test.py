
def dl(data='l', **kwargs):
    print(kwargs)
    print(data)

    print(kwargs)

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



def test(target, *args):
    t = target
    print(t)
    print(args)
    type(args)
    a, b = tuple(args)
    print('first {}'.format(a))
    print('2nd {}'.format(b))

a = ['hi', 'hi2']
test('t', *a)


def test(*args):
    print(args)
    type(args)
    a, b, c = tuple(args)
    print('first {}'.format(a))
    print('2nd {}'.format(b))

a = ['hi', 'hi2']
test(*a, 't')


def test( *args):

    print(args)
    print(type(args))
    a, b = tuple(args)
    print('first {}'.format(a))
    print('2nd {}'.format(b))

a = ['hi', 'hi2']
test('t', ['hi', 'hi2'])
test('t', ['hi', 'hi2'])



f_path = os.path.abspath(os.path.join(root, 'resnet50-19c8e357.pth'))
print(f_path)
print('exist'+str(os.path.exists(f_path))
if os.path.exists(f_path):
    model.load_state_dict(torch.load(f_path), strict=False)
else:
    model.load_state_dict(torch.load(get_model_file('resnet50', root=root)), strict=False)



t = torch.tensor([[1,2],[3,4]])
torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))

import os
path= os.path.expanduser('~/Documents/semantic_seg/models/DANet/encoding/models/resnet50-19c8e357.pth')

model.load_state_dict(torch.load(path), strict=False)
stored = torch.load(path)
params = model.state_dict()

kwargs = {'os':8, 'norm_layer':SyncBatchNorm}
pretrain = get_backbone(backbone, pretrained=True, dilated = True, **kwargs)