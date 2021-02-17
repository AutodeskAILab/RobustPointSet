import os
import sys
import torch
import argparse
import datetime
import logging
import shutil
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




def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points.float())
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
        
    ### Hyper Parameters ###
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    ### Create Dir ###
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'-'+args.model+'-'+args.task
    experiment_dir = Path('log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    ### Log ###
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    
    ### Data Loading ###
    log_string('Load dataset ...')
    TRAIN_DATASET = ModelNetDataLoader(root=args.data_root, 
                                   tasks=args.train_tasks,
                                   labels=args.train_labels,
                                   partition='train',
                                   npoint=args.num_point,      
                                   normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=args.data_root, 
                                   tasks=args.test_tasks,
                                   labels=args.test_labels,
                                   partition='test',
                                   npoint=args.num_point,      
                                   normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    
    ### Model Loading ###
    num_class = 40
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # pointnet_util pointnet_util, 
    shutil.copy('models/%s.py' % args.ults, str(experiment_dir))
    
    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), 
                                    lr=0.01, 
                                    momentum=0.9)

    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []
    
    
    ### Training ###
    logger.info('Start training ...')
    for epoch in range(start_epoch, args.epoch): 
        
        
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        #scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            
            points = points.data.numpy()
            
            """
            # augmentation is not used here
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            """

            
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
        
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        
    
        if (epoch % 2 == 0) or (epoch == args.epoch):
            with torch.no_grad():
                instance_acc, class_acc = test(classifier.eval(), testDataLoader)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
                log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s'% savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
        global_epoch += 1
        
        # adjust lr
        scheduler.step()
        #for param_group in optimizer.param_groups:
        #    print('lr: ' + str(param_group['lr']))
        
    logger.info('End of training...')
    
    
    
    
    
def parseArgs():
    """ Argument parser for configuring model training """
    parser = argparse.ArgumentParser(description='RobustPointSet trainer')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='pointnet_cls', help='point cloud model')
    parser.add_argument('--ults', type=str, default='pointnet_util', help='help functions for point cloud model')
    parser.add_argument('--task', type=str, default='s1', help='Stragety 1 or 2')
    parser.add_argument('--epoch', type=int, default=300, help='training epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=str, default='0', help='gpu device index')
    parser.add_argument('--num_point', type=int, default=2048, help='number of points')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--log_dir', type=str, default=None, help='directory to store training log')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_clip', type=float, default=1e-7)
    parser.add_argument('--normal', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--data_root', type=str, default='data/', help='data directory')
    parser.add_argument('--train_tasks', type=str, nargs='+', required=True, help="List of RobustPointSet files to be trained on")
    parser.add_argument('--test_tasks', type=str, nargs='+', required=True, help="List of RobustPointSet files to be tested on during training")
    return parser.parse_args()



if __name__ == '__main__':
    """
    Available data: original, jitter, translate, missing_part, sparse, rotation, occlusion
    Example command for strategy 1: 
        python train.py --train_tasks train_original.npy --test_tasks test_original.npy
        python train.py --train_tasks train_original.npy --test_tasks test_sparse.npy
    Example command for strategy 2: 
        python train.py --train_tasks train_original.npy train_jitter.npy train_translate.npy train_missing_part.npy train_sparse.npy train_rotation.npy --test_tasks test_occlusion.npy
        python train.py --train_tasks train_original.npy train_translate.npy train_missing_part.npy train_sparse.npy train_rotation.npy train_occlusion.npy --test_tasks test_jitter.npy
    """
    args = parseArgs()
    args.train_labels = ['train_labels.npy']*len(args.train_tasks)
    args.test_labels =['test_labels.npy']*len(args.test_tasks)
    main(args)