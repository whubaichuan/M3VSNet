import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
from torchvision import models
from torchvision import utils as vutils
from torchvision import transforms

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="1,3,5,7:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume',action='store_true', help='continue to train the model') #default=True

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# class vggNet(nn.Module):
#     def __init__(self, pretrained=True):
#         super(vggNet, self).__init__()
#         self.net = models.vgg16(pretrained=True).features.eval()
 
#     def forward(self, x):
#         out = []
#         for i in range(len(self.net)):
#             x = self.net[i](x)
#             #if i in [3, 8, 15, 22, 29]:
#             if i in [3, 8, 15]: #提取1，1/2，1/4的特征图
#                 # print(self.net[i])
#                 out.append(x)
#         return out

# model, optimizer
model = MVSNet(refine=False)
model_feature = vggNet()
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
    model_feature = nn.DataParallel(model_feature)
model.cuda()
model_feature.cuda()
model_loss = mvsnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        #training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            #loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            loss,scalar_outputs,image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                print("save_scalar")
                save_scalars(logger, 'train', scalar_outputs, global_step)
                print("save_images")
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}, GS={}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                     time.time() - start_time,global_step))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    #print("depth_gt_shape:{}".format(depth_gt.shape)) #Batchsize*128*160
    mask_gt = sample_cuda["mask"]
    intrinsics=sample_cuda["intrinsics"]
    extrinsics=sample_cuda["extrinsics"]

    #print(sample_cuda["imgs"].shape) 
    print("Begin forward")
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    photometric_confidence = outputs["photometric_confidence"]


    #提取特征
    # with torch.no_grid()
    with torch.no_grad():
        print("Begin VGG16 extract feature")
        #print(sample_cuda["imgs"].shape) #4*3*3*512*640
        imgs_features = sample_cuda["imgs"].clone()
        imgs_features=torch.unbind(imgs_features,1) # nivews个4*3*512*640

        mean=torch.tensor([[[0.485]],[[0.456]],[[0.406]]],device='cuda')
        std=torch.tensor([[[0.229]],[[0.224]],[[0.225]]],device='cuda')

        imgs_features = [(img-mean)/std for img in imgs_features]
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #print(len(imgs_features)) 3
        #print(imgs_features[0].shape) 4*3*512*640
        outputs_feature = [model_feature(img) for img in imgs_features]

    #print(len(outputs_feature)) #3
    #print(len(outputs_feature[0])) #1
    #print(outputs_feature[0][0].shape) #4*256*128*160
    
    #print(len(outputs_feature)) 3
    #print(outputs_feature[0][0].shape) #4*64*512*640-----3
    #print(outputs_feature[0][1].shape) #4*28*258*320-----8
    #print(outputs_feature[0][2].shape) #4*256*128*160----15


    #depth_est=outputs["refined_depth"]
    #depth_est=depth_est.squeeze(1)

    mask_photometric=photometric_confidence>0.5 #unit8

    mask_final = mask_gt*mask_photometric.float()

    #print(mask_final)
    #print(mask_final.shape)

    #print("depth_est_shape:{}".format(depth_est.shape)) #4*128*160
    #print(depth_est.shape) #B*128*160
    #print(intrinsics.shape) #B*3*3
    #print(sample_cuda["imgs"].shape) #B*3*3*512*640

    #loss = model_loss(depth_est, depth_gt, mask)
    print("Begin calculate loss")
    loss,loss_s,loss_photo,loss_ssim,mask_calculate,mask_num,loss_perceptual,loss_normal,normal_by_depth,error_depth_by_normal,depth_by_normal=model_loss(depth_est,intrinsics,extrinsics,sample_cuda["imgs"],mask_photometric,outputs_feature)
    print("Begin loss backward")
    loss.backward()
    print("Begin optimizer")
    optimizer.step()

    # print(depth_est.shape)
    # print(type(depth_est))
    # print(depth_by_normal.shape)
    # print(type(depth_by_normal))

    #print(mask_num)
    scalar_outputs = {"loss": loss,"loss_s":loss_s,"loss_photo":loss_photo,"loss_ssim":loss_ssim,"loss_perceptual":loss_perceptual,"loss_normal":loss_normal}
    image_outputs = {"depth_est_gt": depth_est * mask_gt, "depth_est_final": depth_est * mask_final,
                    "depth_est":depth_est,
                    "depth_by_normal_final":depth_by_normal * mask_final,
                    "depth_by_normal":depth_by_normal,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask_gt": sample["mask"],
                     "photometric":photometric_confidence,
                     "mask_photometric":mask_photometric.float(),
                     "normal_by_depth":normal_by_depth.permute(0,3,1,2)
                     }
                     #"mask_calculate":mask_calculate}
    if detailed_summary:
        image_outputs["errormap_gt"] = (depth_est - depth_gt).abs() * mask_gt
        image_outputs["errormap_final"] = (depth_est - depth_gt).abs() * mask_final
        image_outputs["errormap_depth_normal_final"]=error_depth_by_normal * mask_final
        #print(depth_est.device)
        #print(depth_gt.device)
        #print(mask_final.device)
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask_final > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 8)
    
    #print("loss:{}".format(loss))
    #print("abs:{}".format(AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)))
    #print("2:{}".format(Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)))
    #print("4:{}".format(Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)))
    #print("8:{}".format(Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)))
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def test_loss_calculate(depth_est, depth_gt, mask_final):
    mask_final=mask_final>0.5
    return F.smooth_l1_loss(depth_est[mask_final], depth_gt[mask_final], size_average=True)


@make_nograd_func
def test_sample(sample, detailed_summary=True):

    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask_gt = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    photometric_confidence = outputs["photometric_confidence"]
    #depth_est=outputs["refined_depth"]
    #depth_est=depth_est.squeeze(1)

    mask_photometric=photometric_confidence>0.5
    mask_final=mask_gt*mask_photometric.float()
    

    loss = test_loss_calculate(depth_est, depth_gt, mask_final)


    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est_gt": depth_est * mask_gt, "depth_est_final": depth_est * mask_final,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask_gt": sample["mask"],
                     "photometric":photometric_confidence,
                     "mask_photometric":mask_photometric.float()}
    if detailed_summary:
        image_outputs["errormap_gt"] = (depth_est - depth_gt).abs() * mask_gt
        image_outputs["errormap_final"] = (depth_est - depth_gt).abs() * mask_final

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask_final > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask_final > 0.5, 8)



    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
