import os
import sys
import torch
import argparse
import importlib
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append("..")
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils import provider

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'log'))



def test(model, loader, num_class=40, vote_num=1):
    
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points.float())
            vote_pool += pred
        pred = vote_pool/vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            cat = int(cat)
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.enabled = False
    
    '''DATA LOADING'''
    TEST_DATASET = ModelNetDataLoader(root=args.data_root, 
                                       tasks=args.test_tasks,
                                       labels=args.test_labels,
                                       partition='test',
                                       npoint=args.num_point,      
                                       normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)

    '''MODEL LOADING'''
    num_class = 40
    files = os.listdir(args.model_dir+'/logs')
    for f in files:
        if f.endswith('txt'):
            model_name = f.split('.')[0]

    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(args.model_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        
        
def parseArgs():
    """ Argument parser for configuring model testing """
    parser = argparse.ArgumentParser(description='RobustPointSet trainer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='pointnet_cls', help='point cloud model')
    parser.add_argument('--ults', type=str, default='pointnet_util', help='help functions for point cloud model')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device index')
    parser.add_argument('--num_point', type=int, default=2048, help='number of points')
    parser.add_argument('--num_votes', type=int, default=1, help='number of time to run testing and doing majority vote')
    parser.add_argument('--normal', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--data_root', type=str, default='data/', help='data directory')
    parser.add_argument('--test_tasks', type=str, nargs='+', required=True, help="List of RobustPointSet files to be tested on during training")
    parser.add_argument('--model_dir', type=str, required=True, help="model checkpoint")
    return parser.parse_args()

if __name__ == '__main__':
    """
    Available data: original, jitter, translate, missing_part, sparse, rotation, occlusion
    Example command for strategy 1 & 2: 
        python test.py --test_tasks test_original.npy --model_dir log/classification/2021-02-17_10-37-pointnet_cls-s1
        python test.py --test_tasks test_rotation.npy --model_dir log/classification/2021-02-17_10-37-pointnet_cls-s1

    """
    args = parseArgs()
    args.test_labels =['test_labels.npy']*len(args.test_tasks)
    main(args) 