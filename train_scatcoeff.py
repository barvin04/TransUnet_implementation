import argparse
import logging
import os
import random
import numpy as np
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tensorboardX import SummaryWriter
from networks.TransUNet_model import TransUNet
from datasets.dataset_us_xray_scat import Ultrasound_dataset, LungXray_dataset, RandomGenerator
from utils import DiceLoss

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Ultrasound', help='name of dataset for training')
parser.add_argument('--list_dir', type=str, help='path to dir where train-val split is stored')
parser.add_argument('--num_classes', type=int,
                    default=2, help='number of classes including background')
parser.add_argument('--models_save_dir', type=str, required=True,
                    help='dir path to save models')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
args = parser.parse_args()

def training_loop(args, model, snapshot_path, dataset_name):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    Dataset = {'Ultrasound': Ultrasound_dataset, 'CovidLungSeg': LungXray_dataset}
    db_train = Dataset[dataset_name](base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(2021 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            scat_mat_batch = sampled_batch['scat_mat']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            scat_mat_batch = scat_mat_batch.cuda()
            outputs = model(image_batch, scat_mat_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        if epoch_num % 20 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":

    dataset_name = args.dataset
    dataset_config = {
        'Ultrasound': {
            'root_path': '/ssd_scratch/cvit/rupraze/data/ultrasound',
            'list_dir': './lists/lists_Ultrasound',
            'num_classes': 2,
        },
        'LungSeg': {
            'root_path': '/ssd_scratch/cvit/rupraze/data/lungs_seg_dataset',
            'list_dir': './lists/lists_CovidLungSeg',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    
    if not os.path.exists(args.models_save_dir):
        os.makedirs(args.models_save_dir)

    net = TransUNet(num_classes=args.num_classes, use_scat_encoder=True).cuda()

    training_loop(args, net, args.models_save_dir, dataset_name)