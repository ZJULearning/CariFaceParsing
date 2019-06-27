###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import torch.nn as nn

import encoding.utils as utils
from encoding.nn import SegmentationLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule, MultiEvalModuleHelen, MultiEvalModuleCityscapes

from option import Options
from PIL import Image

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

palette = [0, 0, 0, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 170, 255, 255, 255, 170, 170, 255, 255, 85, 255, 255, 85, 170, 255, 105, 45, 190,
              170, 105, 255, 170, 255, 85, 255, 170, 85, 255, 85, 170,  85, 255, 85, 0, 255, 85, 255, 0, 85, 255, 85, 85, 0, 85, 255, 85, 85, 255]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def test(args):
    # output folder
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform)
    else:
        testset = get_segmentation_dataset(args.dataset, split='test', mode='test',
                                           transform=input_transform)
    # dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       se_loss = args.se_loss)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])

    evaluator = MultiEvalModuleHelen(model, testset.num_class).cuda()
    evaluator.eval()

    tbar = tqdm(test_data)
    def eval_batch(image, dst, evaluator, eval_mode, hist, names):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            outputs = evaluator.parallel_forward(image)
            batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
            for output, target, name in zip(outputs, targets, names):
                correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target)
                inter, union = utils.batch_intersection_union(
                    output.data.cpu(), target, testset.num_class)
                batch_correct += correct
                batch_label += labeled
                batch_inter += inter
                batch_union += union
                a = target.numpy().flatten()
                b = output.data.cpu()
                _, b = torch.max(b, 1)
                b = b.numpy().flatten()
                n = testset.num_class
                k = (a >= 0) & (a < n)
                hist += np.bincount(n * a[k].astype(int) + b[k], minlength = n ** 2).reshape(n, n)

                output = output.data.cpu().numpy()[0]
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name.split('/')[-1]
                output.save('%s/%s' % (args.save, name))
                output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
            return batch_correct, batch_label, batch_inter, batch_union, hist
        else:
            # test mode, dump the results
            im_paths = dst
            outputs = evaluator.parallel_forward(image)
            predicts = [torch.max(output, 1)[1].cpu().numpy() + testset.pred_offset
                        for output in outputs]
            for predict, impath in zip(predicts, im_paths):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
            # dummy outputs for compatible with eval mode
            return 0, 0, 0, 0

    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    hist =np.zeros((testset.num_class, testset.num_class))

    for i, (image, dst, name) in enumerate(tbar):
        if torch_ver == "0.3":
            image = Variable(image, volatile=True)
            correct, labeled, inter, union, hist = eval_batch(image, dst, evaluator, args.eval, hist, name)
        else:
            with torch.no_grad():
                correct, labeled, inter, union, hist = eval_batch(image, dst, evaluator, args.eval, hist, name)
        if args.eval:
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    mIoUs = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoUs = mIoUs[1:-1]
    print(str(round(np.nanmean(mIoUs) * 100, 2)))
    print(mIoUs)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
