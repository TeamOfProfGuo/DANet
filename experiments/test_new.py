###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
#from model_mapping import rename_weight_for_head

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='deeplab', help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='pascal_voc', help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520, help='base image size')
        parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
        parser.add_argument('--train-split', type=str, default='train', help='dataset train split (default: train)')

        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False, help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False, help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2, help='SE-loss weight (default: 0.2)')
        parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=2, metavar='N', help='batch size for testing')

        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default= True, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        # checking point
        parser.add_argument('--model_path', type=str, default='./runs/pascal_voc/deeplab/resnet50/default/checkpoint.pth',
                            help='put the path to resuming file if needed')

        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False, help='evaluating mIoU')
        parser.add_argument('--export', type=str, default=None, help='put the path to resuming file if needed')
        parser.add_argument('--acc-bn', action='store_true', default= False, help='Re-accumulate BN statistics')
        parser.add_argument('--test-val', action='store_true', default= True, help='generate masks on val set')

        # test option
        parser.add_argument('--test-folder', type=str, default=None, help='path to test image folder')
        # multi grid dilation option
        parser.add_argument("--multi-grid", action="store_true", default=False, help="use multi grid dilation policy")
        parser.add_argument('--multi-dilation', nargs='+', type=int, default=None, help="multi grid dilation list")
        parser.add_argument('--os', type=int, default=8, help='output stride default:8')
        parser.add_argument('--no-deepstem', action="store_true", default=False, help='backbone without deepstem')
        # the parser
        self.parser = parser

    def parse(self):

        args = self.parser.parse_args(['--batch-size', '2',
                                       '--test-val', True,
                                       '--model_path', './runs/pascal_voc/deeplab/resnet50/default/checkpoint.pth'])
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


def main():

    options = Options()
    args = options.parse()

    # data transforms
    input_transform = transform.Compose([transform.ToTensor(),
                                         transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
    elif args.test_val:
        testset = get_dataset(args.dataset, split='val', mode='test', transform=input_transform)
    else:
        testset = get_dataset(args.dataset, split='test', mode='test', transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    testloader = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux, se_loss=args.se_loss,
                                   norm_layer=torch.nn.BatchNorm2d, # if args.acc_bn else SyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size,
                                   #multi_grid=args.multi_grid, multi_dilation=args.multi_dilation, os=args.os,
                                   #no_deepstem=args.no_deepstem
                                   )

    # load model params
    if args.model_path is not None and os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    else:
        raise RuntimeError("=> no checkpoint found")
    print(model)

    #Re - accumulate BN statistics
    if args.acc_bn:
        from encoding.utils.precise_bn import update_bn_stats
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        trainloader = data.DataLoader(ReturnFirstClosure(trainset), batch_size=args.batch_size,
                                      drop_last=True, shuffle=True, **loader_kwargs)
        print('Reseting BN statistics')
        # model.apply(reset_bn_statistics)
        model.cuda()
        update_bn_stats(model, trainloader)

    if args.export:
        torch.save(model.state_dict(), args.export + '.pth')
        return

    metrics = utils.SegmentationMetric(testset.num_class)

    validate(args, model, testloader, metrics)


def validate(args, model, loader, metrics, device, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    #args.outdir = './results/deeplab_resnet50/out'
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples







def test(args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_dataset(args.dataset, split='val', mode='testval',
                              transform=input_transform)
    elif args.test_val:
        testset = get_dataset(args.dataset, split='val', mode='test',
                              transform=input_transform)
    else:
        testset = get_dataset(args.dataset, split='test', mode='test',
                              transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    pretrained = args.resume is None and args.verify is None
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=pretrained)
        model.base_size = args.base_size
        model.crop_size = args.crop_size
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux = args.aux,
                                       se_loss=args.se_loss,
                                       norm_layer=torch.nn.BatchNorm2d if args.acc_bn else SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid,
                                       multi_dilation=args.multi_dilation,
                                       os=args.os,
                                       no_deepstem=args.no_deepstem)

    # resuming checkpoint
    #print("=={}".format(os.path.isfile(args.resume)))
    if args.verify is not None and os.path.isfile(args.verify):
        print("=> loading checkpoint '{}'".format(args.verify))
        model.load_state_dict(torch.load(args.verify))
    elif args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        weights = checkpoint['state_dict']
        model.load_state_dict(weights)
        # strict=False, so that it is compatible with old pytorch saved models
        #model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    elif not pretrained:
        raise RuntimeError ("=> no checkpoint found")
    print(model)

    if args.acc_bn:
        from encoding.utils.precise_bn import update_bn_stats
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        trainloader = data.DataLoader(ReturnFirstClosure(trainset), batch_size=args.batch_size,
                                      drop_last=True, shuffle=True, **loader_kwargs)
        print('Reseting BN statistics')
        #model.apply(reset_bn_statistics)
        model.cuda()
        update_bn_stats(model, trainloader)

    if args.export:  # export the model with updated BN param
        torch.save(model.state_dict(), args.export + '.pth')
        return

    #scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
    scales = [1.0] if args.dataset == 'citys' else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]#, 2.0
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales).cuda()
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        if args.eval:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

    if args.eval:
        print( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))

class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)