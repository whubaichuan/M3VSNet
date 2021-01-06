import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import time
from datasets import ssim


# class CostRegNet(nn.Module):
#     def __init__(self):
#         super(CostRegNet, self).__init__()
#         self.conv0 = ConvBnReLU3D(32, 8)

        
#         self.conv1 = ConvBnReLU3D(8, 8)
#         self.conv2 = ConvBnReLU3D(8, 16, stride=2)

#         self.conv3 = ConvBnReLU3D(16, 16)
#         self.conv4 = ConvBnReLU3D(16, 32, stride=2)
        
#         self.conv5 = ConvBnReLU3D(32, 32)
#         self.conv6 = ConvBnReLU3D(32, 64, stride=2)

#         self.conv7 = ConvBnReLU3D(64, 64)
#         self.conv8 = ConvBnReLU3D(64, 128, stride=2)

#         self.conv9=ConvBnReLU3D(128, 128)

#         self.conv10 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm3d(64),
#             nn.ReLU(inplace=True))
        
#         self.conv11=ConvBnReLU3D(64, 64)

#         self.conv12 = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm3d(32),
#             nn.ReLU(inplace=True))
        
#         self.conv13 = ConvBnReLU3D(32, 32)

#         self.conv14 = nn.Sequential(
#             nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True))

#         self.conv15 = ConvBnReLU3D(16, 16)

#         self.conv16 = nn.Sequential(
#             nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
#             nn.BatchNorm3d(8),
#             nn.ReLU(inplace=True))
        
#         self.conv17 = ConvBnReLU3D(8, 8)
#         self.conv18 = ConvBnReLU3D(8, 8)

#         self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

#     def forward(self, x):
#         conv0 = self.conv0(x)#8
#         conv1=self.conv1(conv0)#8
#         conv2=self.conv2(conv1+conv0)#16
#         conv3=self.conv3(conv2)#16
#         conv4=self.conv4(conv2+conv3)#32
#         conv5=self.conv5(conv4) #32
#         conv6=self.conv6(conv5+conv4)#64
#         conv7=self.conv7(conv6) #64
#         conv8=self.conv8(conv7+conv6)#128
#         conv9=self.conv9(conv8) #128
#         conv10=self.conv10(conv9+conv8)#64

#         conv11=self.conv11(conv10) #64
#         conv12=self.conv12(conv11+conv10+conv7+conv6)#32
#         conv13=self.conv13(conv12)#32
#         conv14=self.conv14(conv13+conv12+conv4+conv5)#16
#         conv15=self.conv15(conv14)#16
#         conv16=self.conv16(conv15+conv14+conv2+conv3)#8
#         conv17=self.conv17(conv16)#8
#         conv18=self.conv18(conv17+conv16+conv0+conv1)

#         x = self.prob(conv18)
#         return x

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        #self.feature = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10= ConvBnReLU(64, 64, 3, 1, 1)
        self.conv11= ConvBnReLU(64, 128,5, 2, 2)

        self.conv12= nn.Conv2d(128, 64, 1, 1, 0)
        self.conv13= nn.Conv2d(64, 32, 1, 1, 0)

        self.conv14= nn.Conv2d(16, 32, 1, 1, 0)

    def forward(self, x):
        #x = self.conv1(self.conv0(x))
        #x = self.conv4(self.conv3(self.conv2(x)))
        #x = self.feature(self.conv6(self.conv5(x)))
        conv2=self.conv2(self.conv1(self.conv0(x)))
        #print(conv2.shape) 16
        conv5=self.conv5(self.conv4(self.conv3(conv2)))
        #print(conv5.shape) 32
        conv8=self.conv8(self.conv7(self.conv6(conv5)))
        #print(conv8.shape) 64
        conv11=self.conv11(self.conv10(self.conv9(conv8)))
        #print(conv11.shape) 128
        conv12=self.conv12(conv11)
        conv12=F.interpolate(conv12,scale_factor=2,mode='bilinear', align_corners=False)
        #print(conv12.shape)
        conv13=self.conv13(conv12+conv8)
        conv13=F.interpolate(conv13,scale_factor=2,mode='bilinear', align_corners=False)
        #print(conv13.shape)
        conv14=self.conv14(conv2)
        conv14=F.interpolate(conv14,scale_factor=1/2,mode='bilinear', align_corners=False)
        #print(conv14.shape)
        feature=conv5+conv13+conv14
        #print(feature.shape)
        return feature


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        #print(conv4.shape)
        #print(self.conv7(x).shape)

        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)

        del conv4,conv2
        torch.cuda.memory_allocated()
        torch.cuda.memory_cached()
        torch.cuda.empty_cache()

        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]

        del features
        torch.cuda.memory_allocated()
        torch.cuda.memory_cached()
        torch.cuda.empty_cache()
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        del src_features
        torch.cuda.memory_allocated()
        torch.cuda.memory_cached()
        torch.cuda.empty_cache()
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        #cost_reg = self.unet3d(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        #print('depth_size:{}'.format(depth.shape))#1*128*160

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            #print(prob_volume_sum4.shape)
            #print(depth_index.shape)
            #print(depth_index)
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

            #print('photometric_confidence_size:{}'.format(photometric_confidence.shape))

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            #print(imgs[0].shape) 1*3*512*640
            #print(depth.shape) 1*128*160
            #print(torch.cat((imgs[0][:,:,1::4,1::4], depth.unsqueeze(1)), 1).shape) 1*4*128*160
            #refined_depth = self.refine_network(torch.cat((imgs[0][:,:,1::4,1::4], depth.unsqueeze(1)), 1))
            refined_depth = self.refine_network(imgs[0][:,:,1::4,1::4],depth.unsqueeze(1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}



def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    width,height=depth_ref.shape[2],depth_ref.shape[1]
    batchsize=depth_ref.shape[0]
    #print("depth_ref_shape:{}".format(depth_ref.shape)) #1*128*160

    y_ref,x_ref=torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                               torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    #print(intrinsics_ref.shape) #4*3*3
    #a=torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * depth_ref.view(batchsize,-1).unsqueeze(1)
    #print(a.shape) 4*3*20480

    #print(width)
    #print(height)
    #print(depth_ref.shape)

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref),
                        torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * depth_ref.view(batchsize,-1).unsqueeze(1))
    #print(x_ref.shape) 20480
    #print(torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).shape) #3*20480
    #print(depth_ref.view(-1).shape)#20480
    #print(xyz_ref.shape) #B*3*20480

    xyz_src = torch.matmul(torch.matmul(extrinsics_src,torch.inverse(extrinsics_ref)),
                        torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize,1,1)),dim=1))[:,:3,:]
    #print(xyz_src.shape)  B*3*20480
    
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src) #B*3*20480
    xy_src = K_xyz_src[:,:2,:] / K_xyz_src[:,2:3,:]
    x_src = xy_src[:,0,:].view([batchsize,height, width])
    y_src = xy_src[:,1,:].view([batchsize,height, width])
    #print(x_src.shape) #B*128*160

    return x_src,y_src

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_x_depth(depth):
    return depth[:,:-1,:]-depth[:,1:,:]

def gradient_y(img):
    return img[:, :, :,:-1] - img[:, :, :, 1:]

def gradient_y_depth(depth):
    return depth[:,:,:-1]-depth[:,:,1:]


def compute_3dpts_batch(pts, intrinsics):
    ## pts is the depth map of rank3 [batch, h, w], intrinsics is in [batch, 4]

    #fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3] 

    #fx, fy, cx, cy = intrinsics[:,0,0], intrinsics[:,1,1], intrinsics[:,0,2], intrinsics[:,1,2] 

    #print(cx.shape) #4

    #pts_shape = pts.get_shape().as_list()
    #pts_3d = tf.zeros(pts.get_shape().as_list()[:2]+[3])

    pts_shape = pts.shape #4*128*160
    batchsize = pts_shape[0]
    height = pts_shape[1]
    width = pts_shape[2]

    #pts_3d = tf.zeros(pts.get_shape().as_list()[:2]+[3])

    y_ref,x_ref=torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=pts.device),
                               torch.arange(0, width, dtype=torch.float32, device=pts.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    # print(y_ref.shape) #128*160
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)


    xyz_ref = torch.matmul(torch.inverse(intrinsics),
                        torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * pts.view(batchsize,-1).unsqueeze(1))

    #print(xyz_ref.shape) 4*3*20480
    xyz_ref = xyz_ref.view(batchsize,3,height,width)
    #print(xyz_ref.shape) 4*3*128*160
    xyz_ref = xyz_ref.permute(0,2,3,1)
    #print(xyz_ref.shape)# 4*128*160*3
    # print(y_ref) 

    #别人实现的
    #y_ref = y_ref.unsqueeze(0).repeat(batchsize,1,1)
    #x_ref = x_ref.unsqueeze(0).repeat(batchsize,1,1)
    #已经被整合进了torch.meshgrid
    # x = tf.range(0, pts.get_shape().as_list()[2])
    # x = tf.cast(x, tf.float32)
    # y = tf.range(0, pts.get_shape().as_list()[1])
    # y = tf.cast(y, tf.float32)
    #多余
    #cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    #cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    #fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    #fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    # pts_x = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[0], 0), [pts_shape[0], 1, 1]) - cx_tile) / fx_tile * pts
    # pts_y = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[1], 0), [pts_shape[0], 1, 1]) - cy_tile) / fy_tile * pts
    # pts_3d = tf.concat([[pts_x], [pts_y], [pts_z]], 0)
    # pts_3d = tf.transpose(pts_3d, perm = [1,2,3,0]) #B*H*W*C

    return xyz_ref

def compute_normal_by_depth(depth_est, ref_intrinsics,nei):

    ## mask is used to filter the background with infinite depth
    #mask = tf.greater(depth_map, tf.zeros(depth_map.get_shape().as_list())) #我这里好像不存在depth<0的点

    #kitti_shape = depth_map.get_shape().as_list()
    depth_est_shape = depth_est.shape #4*128*160
    batchsize = depth_est_shape[0]
    height = depth_est_shape[1]
    width = depth_est_shape[2]

    pts_3d_map = compute_3dpts_batch(depth_est, ref_intrinsics) #4*128*160*3
    pts_3d_map = pts_3d_map.contiguous()

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0 #因为是求向量，所以不用除以相邻两点之间的距离
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    #pix_num = kitti_shape[0] * (kitti_shape[1]-2*nei) * (kitti_shape[2]-2*nei)
    pix_num=batchsize*(height-2*nei)*(width-2*nei)
    #print(pix_num)
    #print(diff_x0.shape)
    diff_x0 = diff_x0.view(pix_num, 3)
    diff_y0 = diff_y0.view(pix_num, 3)
    diff_x1 = diff_x1.view(pix_num, 3)
    diff_y1 = diff_y1.view(pix_num, 3)
    diff_x0y0 = diff_x0y0.view(pix_num, 3)
    diff_x0y1 = diff_x0y1.view(pix_num, 3)
    diff_x1y0 = diff_x1y0.view(pix_num, 3)
    diff_x1y1 = diff_x1y1.view(pix_num, 3)

    ## calculate normal by cross product of two vectors
    normals0 = F.normalize(torch.cross(diff_x1, diff_y1)) #* tf.tile(normals0_mask[:, None], [1,3]) tf.tile=.repeat
    normals1 = F.normalize(torch.cross(diff_x0, diff_y0)) #* tf.tile(normals1_mask[:, None], [1,3])
    normals2 = F.normalize(torch.cross(diff_x0y1, diff_x0y0)) #* tf.tile(normals2_mask[:, None], [1,3])
    normals3 = F.normalize(torch.cross(diff_x1y0, diff_x1y1)) #* tf.tile(normals3_mask[:, None], [1,3])
    
    normal_vector = normals0+normals1+normals2+normals3
    #normal_vector = tf.reduce_sum(tf.concat([[normals0], [normals1], [normals2], [normals3]], 0),0)
    #normal_vector = F.normalize(normals0)
    normal_vector = F.normalize(normal_vector)
    #normal_map = tf.reshape(tf.squeeze(normal_vector), [kitti_shape[0]]+[kitti_shape[1]-2*nei]+[kitti_shape[2]-2*nei]+[3])
    normal_map = normal_vector.view(batchsize,height-2*nei,width-2*nei,3)

    #对于depth小于0的点，不计算normal
    #normal_map *= tf.tile(tf.expand_dims(tf.cast(mask[:, nei:-nei, nei:-nei], tf.float32), -1), [1,1,1,3])

    #normal_map = tf.pad(normal_map, [[0,0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
    normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)

    #print(normal_map.shape) #4*128*160*3
    #print(normal_map[0,:,:,0])

    return normal_map


def compute_depth_by_normal(depth_map, normal_map, intrinsics, tgt_image, nei=1):

    depth_init = depth_map.clone()

    d2n_nei = 1 #normal_depth转化的时候的空边
    depth_map = depth_map[:,d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei)]
    normal_map = normal_map[:,d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei), :]

    #depth_dims = depth_map.get_shape().as_list()
    depth_map_shape = depth_map.shape
    batchsize = depth_map_shape[0] #4
    height = depth_map_shape[1] # 126
    width = depth_map_shape[2] # 158

    # x_coor = tf.range(nei, depth_dims[2]+nei)
    # y_coor = tf.range(nei, depth_dims[1]+nei)
    # x_ctr, y_ctr = tf.meshgrid(x_coor, y_coor)
    y_ctr,x_ctr=torch.meshgrid([torch.arange(d2n_nei, height+d2n_nei, dtype=torch.float32, device=normal_map.device),
                               torch.arange(d2n_nei, width+d2n_nei, dtype=torch.float32, device=normal_map.device)])
    y_ctr, x_ctr = y_ctr.contiguous(), x_ctr.contiguous()

    #x_ctr = tf.cast(x_ctr, tf.float32)
    #y_ctr = tf.cast(y_ctr, tf.float32)
    # x_ctr_tile = tf.tile(tf.expand_dims(x_ctr, 0), [depth_dims[0], 1, 1])
    # y_ctr_tile = tf.tile(tf.expand_dims(y_ctr, 0), [depth_dims[0], 1, 1])
    x_ctr_tile = x_ctr.unsqueeze(0).repeat(batchsize,1,1) #B*height*width
    y_ctr_tile = y_ctr.unsqueeze(0).repeat(batchsize,1,1)

    x0 = x_ctr_tile-d2n_nei
    y0 = y_ctr_tile-d2n_nei
    x1 = x_ctr_tile+d2n_nei
    y1 = y_ctr_tile+d2n_nei
    normal_x = normal_map[:,:,:,0]
    normal_y = normal_map[:,:,:,1]
    normal_z = normal_map[:,:,:,2]

    #fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
    fx, fy, cx, cy = intrinsics[:,0,0], intrinsics[:,1,1], intrinsics[:,0,2], intrinsics[:,1,2] 

    # cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, depth_dims[1], depth_dims[2]])
    # cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, depth_dims[1], depth_dims[2]])
    # fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, depth_dims[1], depth_dims[2]])
    # fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, depth_dims[1], depth_dims[2]])
    cx_tile = cx.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
    cy_tile = cy.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
    fx_tile = fx.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
    fy_tile = fy.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
    #print(cx.shape)
    #print(cx_tile.shape)
    #print(cx_tile)2

    numerator = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0 = (x0 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_y0 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1 = (x1 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_y1 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0y0 = (x0 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0y1 = (x0 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1y0 = (x1 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1y1 = (x1 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z


    mask_x0 = denominator_x0 == 0
    denominator_x0 =denominator_x0+1e-3 * mask_x0.float()
    mask_y0 = denominator_y0 == 0
    denominator_y0 =denominator_y0+1e-3 * mask_y0.float()
    mask_x1 = denominator_x1 == 0
    denominator_x1 =denominator_x1+1e-3 * mask_x1.float()
    mask_y1 = denominator_y1 == 0
    denominator_y1 =denominator_y1+1e-3 * mask_y1.float()
    mask_x0y0 = denominator_x0y0 == 0
    denominator_x0y0 =denominator_x0y0 + 1e-3 * mask_x0y0.float()
    mask_x0y1 = denominator_x0y1 == 0
    denominator_x0y1 =denominator_x0y1+ 1e-3 * mask_x0y1.float()
    mask_x1y0 = denominator_x1y0 ==0
    denominator_x1y0 =denominator_x1y0+ 1e-3 * mask_x1y0.float()
    mask_x1y1 = denominator_x1y1 ==0
    denominator_x1y1 =denominator_x1y1+ 1e-3 * mask_x1y1.float()

    # depth_map_x0 = (F.sigmoid(numerator / denominator_x0 - 1.0) * 2.0 + 4.0) * depth_map
    # depth_map_y0 = (F.sigmoid(numerator / denominator_y0 - 1.0) * 2.0 + 4.0) * depth_map
    # depth_map_x1 = (F.sigmoid(numerator / denominator_x1 - 1.0) * 2.0 + 4.0) * depth_map
    # depth_map_y1 = (F.sigmoid(numerator / denominator_y1 - 1.0) * 2.0 + 4.0) * depth_map

    depth_map_x0 = numerator / denominator_x0 * depth_map
    depth_map_y0 = numerator / denominator_y0 * depth_map
    depth_map_x1 = numerator / denominator_y0 * depth_map
    depth_map_y1 = numerator / denominator_y0 * depth_map
    depth_map_x0y0 = numerator / denominator_x0y0 * depth_map
    depth_map_x0y1 = numerator / denominator_x0y1 * depth_map
    depth_map_x1y0 = numerator / denominator_x1y0 * depth_map
    depth_map_x1y1 = numerator / denominator_x1y1 * depth_map

    #print(depth_map_x0.shape) #4*126*158

    depth_x0 = depth_init
    depth_x0[:,d2n_nei:-(d2n_nei),:-(2*d2n_nei)] = depth_map_x0
    depth_y0 = depth_init
    depth_y0[:,0:-(2*d2n_nei),d2n_nei:-(d2n_nei)] = depth_map_y0
    depth_x1 = depth_init
    depth_x1[:,d2n_nei:-(d2n_nei),2*d2n_nei:] = depth_map_x1
    depth_y1 = depth_init
    depth_y1[:,2*d2n_nei:,d2n_nei:-(d2n_nei)] = depth_map_y1
    depth_x0y0 = depth_init
    depth_x0y0[:,0:-(2*d2n_nei),0:-(2*d2n_nei)] = depth_map_x0y0
    depth_x1y0 = depth_init
    depth_x1y0[:,0:-(2*d2n_nei),2*d2n_nei:] = depth_map_x1y0
    depth_x0y1 = depth_init
    depth_x0y1[:,2*d2n_nei:,0:-(2*d2n_nei)] = depth_map_x0y1
    depth_x1y1 = depth_init
    depth_x1y1[:,2*d2n_nei:,2*d2n_nei:] = depth_map_x1y1

#--------------------计算权重--------------------------
    tgt_image = tgt_image.permute(0,2,3,1)
    tgt_image = tgt_image.contiguous() #4*128*160*3


    #print(depth_map_x0.shape)  #4*124*156
    #normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)

    img_grad_x0=tgt_image[:,d2n_nei:-d2n_nei,:-2*d2n_nei,:]-tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    #print(img_grad_x0.shape) #4*126*158*3
    img_grad_x0=F.pad(img_grad_x0,(0,0,0,2*d2n_nei,d2n_nei,d2n_nei),"constant",1e-3)
    img_grad_y0=tgt_image[:,:-2*d2n_nei,d2n_nei:-d2n_nei,:]-tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_y0=F.pad(img_grad_y0,(0,0,d2n_nei,d2n_nei,0,2*d2n_nei),"constant",1e-3)
    img_grad_x1=tgt_image[:,d2n_nei:-d2n_nei,2*d2n_nei:,:] -tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_x1=F.pad(img_grad_x1,(0,0,2*d2n_nei,0,d2n_nei,d2n_nei),"constant",1e-3)
    img_grad_y1=tgt_image[:,2*d2n_nei:,d2n_nei:-d2n_nei,:] -tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_y1=F.pad(img_grad_y1,(0,0,d2n_nei,d2n_nei,2*d2n_nei,0),"constant",1e-3)

    img_grad_x0y0 = tgt_image[:,:-2*d2n_nei,:-2*d2n_nei,:]-tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_x0y0 = F.pad(img_grad_x0y0,(0,0,0,2*d2n_nei,0,2*d2n_nei),"constant",1e-3)
    img_grad_x1y0 = tgt_image[:,:-2*d2n_nei,2*d2n_nei:,:] -tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_x1y0 = F.pad(img_grad_x1y0,(0,0,2*d2n_nei,0,0,2*d2n_nei),"constant",1e-3)
    img_grad_x0y1 = tgt_image[:,2*d2n_nei:,:-2*d2n_nei,:] -tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_x0y1 = F.pad(img_grad_x0y1,(0,0,0,2*d2n_nei,2*d2n_nei,0),"constant",1e-3)
    img_grad_x1y1 = tgt_image[:,2*d2n_nei:,2*d2n_nei:,:]  -tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei,:]
    img_grad_x1y1 = F.pad(img_grad_x1y1,(0,0,2*d2n_nei,0,2*d2n_nei,0),"constant",1e-3)

    #print(img_grad_x0.shape) #4*128*160*3

    alpha = 0.1
    weights_x0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0),3))
    weights_y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_y0),3))
    weights_x1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1),3))
    weights_y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_y1),3))

    weights_x0y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0y0),3))
    weights_x1y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1y0),3))
    weights_x0y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0y1),3))
    weights_x1y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1y1),3))

    #print(weights_x0.shape)    #4*128*160
    weights_sum = torch.sum(torch.stack((weights_x0,weights_y0,weights_x1,weights_y1,weights_x0y0,weights_x1y0,weights_x0y1,weights_x1y1),0),0)
    
    #print(weights.shape) 4*128*160
    weights = torch.stack((weights_x0,weights_y0,weights_x1,weights_y1,weights_x0y0,weights_x1y0,weights_x0y1,weights_x1y1),0)/weights_sum
    depth_map_avg = torch.sum(torch.stack((depth_x0,depth_y0,depth_x1,depth_y1,depth_x0y0,depth_x1y0,depth_x0y1,depth_x1y1),0)*weights,0)
#--------------------计算权重--------------------------
    #depth_map_avg = (depth_x0+depth_y0+depth_x1+depth_y1+depth_x0y0+depth_x1y0+depth_x0y1+depth_x1y1)/8



    # img_grad_x0 = tf.pad(tgt_image[:,:,nei:,:] - tgt_image[:,:,:-1*nei,:],[[0,0],[0,0],[0,nei],[0,0]])
    # img_grad_y0 = tf.pad(tgt_image[:,nei:,:,:] - tgt_image[:,:-1*nei,:,:],[[0,0],[0,nei],[0,0],[0,0]])
    # img_grad_x1 = tf.pad(tgt_image[:,:,2*nei:,:] - tgt_image[:,:,nei:-1*nei,:],[[0,0],[0,0],[2*nei,0],[0,0]])
    # img_grad_y1 = tf.pad(tgt_image[:,2*nei:,:,:] - tgt_image[:,nei:-1*nei,:,:],[[0,0],[2*nei,0],[0,0],[0,0]])

    # alpha = 0.1
    # weights_x0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x0), 3))
    # weights_y0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y0), 3))
    # weights_x1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x1), 3))
    # weights_y1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y1), 3))
    # weights = tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]) / \
    #             tf.reduce_sum(tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]), 0)

    # depth_map_avg = tf.reduce_sum(tf.stack([depth_map_x0, depth_map_y0, depth_map_x1, depth_map_y1])* weights, 0)


####-----------------add----nei--------------####
    ## depth is of rank 3 [batch, height, width]
    ## normal_map batch*height*width*3
    ## intrinsics batch*3*3

#     depth_init = depth_map.clone()



#     d2n_nei = 1 #normal_depth转化的时候的空边
#     depth_map = depth_map[:,d2n_nei+nei:-(d2n_nei+nei), d2n_nei+nei:-(d2n_nei+nei)]
#     normal_map = normal_map[:,d2n_nei+nei:-(d2n_nei+nei), d2n_nei+nei:-(d2n_nei+nei), :]

#     #depth_dims = depth_map.get_shape().as_list()
#     depth_map_shape = depth_map.shape
#     batchsize = depth_map_shape[0] #4
#     height = depth_map_shape[1] # 124
#     width = depth_map_shape[2] # 156

#     # x_coor = tf.range(nei, depth_dims[2]+nei)
#     # y_coor = tf.range(nei, depth_dims[1]+nei)
#     # x_ctr, y_ctr = tf.meshgrid(x_coor, y_coor)
#     y_ctr,x_ctr=torch.meshgrid([torch.arange(nei, height+nei, dtype=torch.float32, device=normal_map.device),
#                                torch.arange(nei, width+nei, dtype=torch.float32, device=normal_map.device)])
#     y_ctr, x_ctr = y_ctr.contiguous(), x_ctr.contiguous()

#     #x_ctr = tf.cast(x_ctr, tf.float32)
#     #y_ctr = tf.cast(y_ctr, tf.float32)
#     # x_ctr_tile = tf.tile(tf.expand_dims(x_ctr, 0), [depth_dims[0], 1, 1])
#     # y_ctr_tile = tf.tile(tf.expand_dims(y_ctr, 0), [depth_dims[0], 1, 1])
#     x_ctr_tile = x_ctr.unsqueeze(0).repeat(batchsize,1,1) #B*height*width
#     y_ctr_tile = y_ctr.unsqueeze(0).repeat(batchsize,1,1)

#     x0 = x_ctr_tile-nei
#     y0 = y_ctr_tile-nei
#     x1 = x_ctr_tile+nei
#     y1 = y_ctr_tile+nei
#     normal_x = normal_map[:,:,:,0]
#     normal_y = normal_map[:,:,:,1]
#     normal_z = normal_map[:,:,:,2]

#     #fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
#     fx, fy, cx, cy = intrinsics[:,0,0], intrinsics[:,1,1], intrinsics[:,0,2], intrinsics[:,1,2] 

#     # cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, depth_dims[1], depth_dims[2]])
#     # cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, depth_dims[1], depth_dims[2]])
#     # fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, depth_dims[1], depth_dims[2]])
#     # fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, depth_dims[1], depth_dims[2]])
#     cx_tile = cx.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
#     cy_tile = cy.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
#     fx_tile = fx.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
#     fy_tile = fy.unsqueeze(-1).unsqueeze(-1).repeat(1,height,width)
#     #print(cx.shape)
#     #print(cx_tile.shape)
#     #print(cx_tile)

#     numerator = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x0 = (x0 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_y0 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x1 = (x1 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_y1 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x0y0 = (x0 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x0y1 = (x0 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x1y0 = (x1 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
#     denominator_x1y1 = (x1 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z

#     # mask_x0 = 1e-3 * (tf.cast(tf.equal(denominator_x0, tf.zeros(denominator_x0.get_shape().as_list())), tf.float32))
#     # denominator_x0 += mask_x0
#     # mask_y0 = 1e-3 * (tf.cast(tf.equal(denominator_y0, tf.zeros(denominator_y0.get_shape().as_list())), tf.float32))
#     # denominator_y0 += mask_y0
#     # mask_x1 = 1e-3 * (tf.cast(tf.equal(denominator_x1, tf.zeros(denominator_x1.get_shape().as_list())), tf.float32))
#     # denominator_x1 += mask_x1
#     # mask_y1 = 1e-3 * (tf.cast(tf.equal(denominator_y1, tf.zeros(denominator_y1.get_shape().as_list())), tf.float32))
#     # denominator_y1 += mask_y1
#     # mask_x0y0 = 1e-3 * (tf.cast(tf.equal(denominator_x0y0, tf.zeros(denominator_x0y0.get_shape().as_list())), tf.float32))
#     # denominator_x0y0 += mask_x0y0
#     # mask_x0y1 = 1e-3 * (tf.cast(tf.equal(denominator_x0y1, tf.zeros(denominator_x0y1.get_shape().as_list())), tf.float32))
#     # denominator_x0y1 += mask_x0y1
#     # mask_x1y0 = 1e-3 * (tf.cast(tf.equal(denominator_x1y0, tf.zeros(denominator_x1y0.get_shape().as_list())), tf.float32))
#     # denominator_x1y0 += mask_x1y0
#     # mask_x1y1 = 1e-3 * (tf.cast(tf.equal(denominator_x1y1, tf.zeros(denominator_x1y1.get_shape().as_list())), tf.float32))
#     # denominator_x1y1 += mask_x1y1

#     mask_x0 = denominator_x0 == 0
#     denominator_x0 =denominator_x0+1e-3 * mask_x0.float()
#     mask_y0 = denominator_y0 == 0
#     denominator_y0 =denominator_y0+1e-3 * mask_y0.float()
#     mask_x1 = denominator_x1 == 0
#     denominator_x1 =denominator_x1+1e-3 * mask_x1.float()
#     mask_y1 = denominator_y1 == 0
#     denominator_y1 =denominator_y1+1e-3 * mask_y1.float()
#     mask_x0y0 = denominator_x0y0 == 0
#     denominator_x0y0 =denominator_x0y0 + 1e-3 * mask_x0y0.float()
#     mask_x0y1 = denominator_x0y1 == 0
#     denominator_x0y1 =denominator_x0y1+ 1e-3 * mask_x0y1.float()
#     mask_x1y0 = denominator_x1y0 ==0
#     denominator_x1y0 =denominator_x1y0+ 1e-3 * mask_x1y0.float()
#     mask_x1y1 = denominator_x1y1 ==0
#     denominator_x1y1 =denominator_x1y1+ 1e-3 * mask_x1y1.float()

#     # depth_map_x0 = (F.sigmoid(numerator / denominator_x0 - 1.0) * 2.0 + 4.0) * depth_map
#     # depth_map_y0 = (F.sigmoid(numerator / denominator_y0 - 1.0) * 2.0 + 4.0) * depth_map
#     # depth_map_x1 = (F.sigmoid(numerator / denominator_x1 - 1.0) * 2.0 + 4.0) * depth_map
#     # depth_map_y1 = (F.sigmoid(numerator / denominator_y1 - 1.0) * 2.0 + 4.0) * depth_map

#     depth_map_x0 = numerator / denominator_x0 * depth_map
#     depth_map_y0 = numerator / denominator_y0 * depth_map
#     depth_map_x1 = numerator / denominator_y0 * depth_map
#     depth_map_y1 = numerator / denominator_y0 * depth_map
#     depth_map_x0y0 = numerator / denominator_x0y0 * depth_map
#     depth_map_x0y1 = numerator / denominator_x0y1 * depth_map
#     depth_map_x1y0 = numerator / denominator_x1y0 * depth_map
#     depth_map_x1y1 = numerator / denominator_x1y1 * depth_map

#     ## fill the peripheral part (nei) of newly generated with 1e6
#     # padding_x0 = [[0,0], [d2n_nei+nei, d2n_nei+nei], [d2n_nei, d2n_nei+2*nei]]
#     # padding_y0 = [[0,0], [d2n_nei, d2n_nei+2*nei], [d2n_nei+nei, d2n_nei+nei]]
#     # padding_x1 = [[0,0], [d2n_nei+nei, d2n_nei+nei], [d2n_nei+2*nei, d2n_nei]]
#     # padding_y1 = [[0,0], [d2n_nei+2*nei, d2n_nei], [d2n_nei+nei, d2n_nei+nei]]
#     # padding_x0y0 = [[0,0], [0, 2*nei], [0, 2*nei]]
#     # padding_x1y0 = [[0,0], [0, 2*nei], [2*nei, 0]]
#     # padding_x0y1 = [[0,0], [2*nei, 0], [0, 2*nei]]
#     # padding_x1y1 = [[0,0], [2*nei, 0], [2*nei, 0]]

#     #normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)
#     # depth_map_x0 = tf.pad(depth_map_x0-1e3, padding_x0)+1e3
#     # depth_map_y0 = tf.pad(depth_map_y0-1e3, padding_y0)+1e3
#     # depth_map_x1 = tf.pad(depth_map_x1-1e3, padding_x1)+1e3
#     # depth_map_y1 = tf.pad(depth_map_y1-1e3, padding_y1)+1e3



#     tgt_image = tgt_image.permute(0,2,3,1)
#     tgt_image = tgt_image.contiguous() #4*128*160*3



#     tgt_image = tgt_image[:,d2n_nei:-d2n_nei,d2n_nei:-d2n_nei] #4*126*158*3
 
#     #print(depth_map_x0.shape)  #4*124*156
#     #normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)

#     img_grad_x0=F.pad(tgt_image[:,nei:-nei,:-2*nei,:]-tgt_image[:,nei:-nei,nei:-nei,:],(0,0,),"constant", 0)

#     img_grad_y0=tgt_image[:,:-2*nei,nei:-nei,:]-tgt_image[:,nei:-nei,nei:-nei,:]
#     img_grad_x1=tgt_image[:,nei:-nei,2*nei:,:] -tgt_image[:,nei:-nei,nei:-nei,:]
#     img_grad_y1=tgt_image[:,2*nei:,nei:-nei,:] -tgt_image[:,nei:-nei,nei:-nei,:]

#     img_grad_x0y0 = tgt_image[:,:-2*nei,:-2*nei,:]-tgt_image[:,nei:-nei,nei:-nei,:]
#     img_grad_x1y0 = tgt_image[:,:-2*nei,2*nei:,:]-tgt_image[:,nei:-nei,nei:-nei,:]
#     img_grad_x0y1 = tgt_image[:,2*nei:,:-2*nei,:]-tgt_image[:,nei:-nei,nei:-nei,:]
#     img_grad_x1y1 = tgt_image[:,2*nei:,2*nei:,:]-tgt_image[:,nei:-nei,nei:-nei,:]


#     alpha = 0.1
#     weights_x0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0),3))
#     weights_y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_y0),3))
#     weights_x1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1),3))
#     weights_y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_y1),3))

#     weights_x0y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0y0),3))
#     weights_x1y0 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1y0),3))
#     weights_x0y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x0y1),3))
#     weights_x1y1 = torch.exp(-1*alpha*torch.mean(torch.abs(img_grad_x1y1),3))

#     #print(weights_x0.shape)    #4*124*156
#     weights = weight_y


# ####------------new----------
#     depth_x0 = depth_init
#     depth_x0[:,d2n_nei+nei:-(d2n_nei+nei),d2n_nei:-(d2n_nei+2*nei)] = depth_map_x0
#     #torch.sum(depth_x0).backward() 
#     ######错误示范
#     depth_y0 = depth_init
#     depth_y0[:,d2n_nei:-(d2n_nei+2*nei),d2n_nei+nei:-(d2n_nei+nei)] = depth_map_y0
#     depth_x1 = depth_init
#     depth_x1[:,d2n_nei+nei:-(d2n_nei+nei),d2n_nei+2*nei:-(d2n_nei)] = depth_map_x1
#     depth_y1 = depth_init
#     depth_y1[:,d2n_nei+2*nei:-(d2n_nei),d2n_nei+nei:-(d2n_nei+nei)] = depth_map_y1
#     depth_x0y0 = depth_init
#     depth_x0y0[:,0:-(2*nei+2*d2n_nei),0:-(2*nei+2*d2n_nei)] = depth_map_x0y0
#     depth_x1y0 = depth_init
#     depth_x1y0[:,0:-(2*nei+2*d2n_nei),2*nei+2*d2n_nei:] = depth_map_x1y0
#     depth_x0y1 = depth_init
#     depth_x0y1[:,2*nei+2*d2n_nei:,0:-(2*nei+2*d2n_nei)] = depth_map_x0y1
#     depth_x1y1 = depth_init
#     depth_x1y1[:,2*nei+2*d2n_nei:,2*nei+2*d2n_nei:] = depth_map_x1y1
# ####------------new----------




# ###----------old------------
#     # depth_x0 = depth_init
#     # depth_x0[:,d2n_nei+nei:-(d2n_nei+nei),d2n_nei:-(d2n_nei+2*nei)] = depth_map_x0
#     # #torch.sum(depth_x0).backward() 
#     # ######错误示范
#     # depth_y0 = depth_init
#     # depth_y0[:,d2n_nei:-(d2n_nei+2*nei),d2n_nei+nei:-(d2n_nei+nei)] = depth_map_y0
#     # depth_x1 = depth_init
#     # depth_x1[:,d2n_nei+nei:-(d2n_nei+nei),d2n_nei+2*nei:-(d2n_nei)] = depth_map_x1
#     # depth_y1 = depth_init
#     # depth_y1[:,d2n_nei+2*nei:-(d2n_nei),d2n_nei+nei:-(d2n_nei+nei)] = depth_map_y1
#     # depth_x0y0 = depth_init
#     # depth_x0y0[:,0:-(2*nei+2*d2n_nei),0:-(2*nei+2*d2n_nei)] = depth_map_x0y0
#     # depth_x1y0 = depth_init
#     # depth_x1y0[:,0:-(2*nei+2*d2n_nei),2*nei+2*d2n_nei:] = depth_map_x1y0
#     # depth_x0y1 = depth_init
#     # depth_x0y1[:,2*nei+2*d2n_nei:,0:-(2*nei+2*d2n_nei)] = depth_map_x0y1
#     # depth_x1y1 = depth_init
#     # depth_x1y1[:,2*nei+2*d2n_nei:,2*nei+2*d2n_nei:] = depth_map_x1y1
# ###----------old------------

#     # img_grad_x0 = tf.pad(tgt_image[:,:,nei:,:] - tgt_image[:,:,:-1*nei,:],[[0,0],[0,0],[0,nei],[0,0]])
#     # img_grad_y0 = tf.pad(tgt_image[:,nei:,:,:] - tgt_image[:,:-1*nei,:,:],[[0,0],[0,nei],[0,0],[0,0]])
#     # img_grad_x1 = tf.pad(tgt_image[:,:,2*nei:,:] - tgt_image[:,:,nei:-1*nei,:],[[0,0],[0,0],[2*nei,0],[0,0]])
#     # img_grad_y1 = tf.pad(tgt_image[:,2*nei:,:,:] - tgt_image[:,nei:-1*nei,:,:],[[0,0],[2*nei,0],[0,0],[0,0]])

#     # alpha = 0.1
#     # weights_x0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x0), 3))
#     # weights_y0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y0), 3))
#     # weights_x1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x1), 3))
#     # weights_y1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y1), 3))
#     # weights = tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]) / \
#     #             tf.reduce_sum(tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]), 0)

#     # depth_map_avg = tf.reduce_sum(tf.stack([depth_map_x0, depth_map_y0, depth_map_x1, depth_map_y1])* weights, 0)
#     depth_map_avg = (depth_x0+depth_y0+depth_x1+depth_y1+depth_x0y0+depth_x1y0+depth_x0y1+depth_x1y1)/8

    return depth_map_avg

def mvsnet_loss(depth_est_1,intrinsics,extrinsics,imgs,mask_photometric,outputs_feature):

    intrinsics=torch.unbind(intrinsics,1)
    extrinsics=torch.unbind(extrinsics,1)
    imgs = torch.unbind(imgs, 1)
    ref_intrinsics, src_intrinsics=intrinsics[0],intrinsics[1:]
    ref_extrinsics, src_extrinsics=extrinsics[0],extrinsics[1:]
    ref_vgg_feature, src_vgg_feature=outputs_feature[0],outputs_feature[1:]
    ref_img, src_img = imgs[0],imgs[1:]
    ref_color=ref_img[:,:,1::4,1::4] #B*C*128*160
    #print(ref_extrinsics.shape)#B*4*4
    #print(ref_intrinsics.shape)#B*3*3

    #depth_to_normal
    #print(depth_est.shape) #4*128*160
    nei=1 #与normal_to_depth相关联，谨慎更改
    normal_by_depth = compute_normal_by_depth(depth_est_1,ref_intrinsics,nei)

    #normal_to_depth
    depth_by_normal = compute_depth_by_normal(depth_est_1,normal_by_depth,ref_intrinsics,ref_color)

    #print(depth_by_normal.shape) 4*128*160

    loss_normal=0
    
    if F.smooth_l1_loss(depth_by_normal,depth_est_1).nelement()==0:
        loss_normal+=torch.tensor(0.)
    else:
        loss_normal+=F.smooth_l1_loss(depth_by_normal,depth_est_1)

    depth_est=depth_est_1
    #depth_est=depth_by_normal

    # del depth_by_normal,loss_normal
    # torch.cuda.memory_allocated()
    # torch.cuda.memory_cached()
    # torch.cuda.empty_cache()

    loss_s=0
    loss_photo=0
    loss_ssim=0
    loss_perceptual=0

    #loss_s
    #print("ref_color_shape:{}".format(ref_color.shape)) #batchsize*3*128*160
    
    
    ref_color_dx=gradient_x(ref_color)
    ref_color_dy=gradient_y(ref_color)
    depth_dx=gradient_x_depth(depth_est)
    depth_dy=gradient_y_depth(depth_est)
    weight_x=torch.exp(-torch.mean(torch.abs(ref_color_dx),1))
    weight_y=torch.exp(-torch.mean(torch.abs(ref_color_dy),1))
    smooth_x=depth_dx*weight_x
    smooth_y=depth_dy*weight_y
    loss_s+=torch.mean(torch.abs(smooth_x))+torch.mean(torch.abs(smooth_y))

    ref_color_d2x=gradient_x(ref_color_dx)
    ref_color_d2y=gradient_y(ref_color_dy)
    depth_d2x = gradient_x_depth(depth_dx)
    depth_d2y = gradient_y_depth(depth_dy)

    weight_x2=torch.exp(-torch.mean(torch.abs(ref_color_d2x),1))
    weight_y2=torch.exp(-torch.mean(torch.abs(ref_color_d2y),1))

    smooth_x2=depth_d2x*weight_x2
    smooth_y2=depth_d2y*weight_y2

    loss_s+=torch.mean(torch.abs(smooth_x2))+torch.mean(torch.abs(smooth_y2))

    #loss_photo & loss_perceptual
    #print(depth_est.shape)#B*128*160
    width, height = depth_est.shape[2], depth_est.shape[1]
    batchsize=depth_est.shape[0]

    for i in range(len(src_img)):
        #intrincs需要缩小4倍吗？之前好像在哪里看到过，现在暂时找不到了，邮件有回复
        #----------------------------question---------------------# 

        x_src,y_src=project_with_depth(depth_est, ref_intrinsics, ref_extrinsics, src_intrinsics[i], src_extrinsics[i])
        src_color=src_img[i][:,:, 1::4, 1::4]
        grid=torch.stack((x_src.view(batchsize,-1)/((width - 1) / 2) - 1,y_src.view(batchsize,-1)/((height - 1) / 2) - 1),2).unsqueeze(0)
        
        # #print(grid.shape) 1*1*20480*2

        #print(src_vgg_feature[i][0].shape) #4*256*128*160  
        ##------------------------------------------------##
        #因为每一层卷积后面跟着relu,所以所有的值都是大于等于0的

        # print(len(src_vgg_feature)) 2
        # print(len(src_vgg_feature[0])) 5
        # print(src_vgg_feature[0][0].shape) 4*64*512*640
        # print(src_vgg_feature[0][1].shape)  4*128*256*320
        # print(src_vgg_feature[0][2].shape)  4*256*128*160
        # print(src_vgg_feature[0][3].shape)  4*512*64*80
        # print(src_vgg_feature[0][4].shape)  4*512*32*40
        #print(ref_intrinsics.shape) 4*3*3

    #    #3 内存爆了
    #     height_perceptual=int(height*4)
    #     width_perceptual=int(width*4)
    #     ref_intrinsics_perceptual=ref_intrinsics
    #     ref_intrinsics_perceptual[:,:2,:]*=4
    #     src_intrinsics_perceptual=src_intrinsics[i]
    #     src_intrinsics_perceptual[:,:2,:]*=4

    #     depth_est_perceptual=F.interpolate(depth_est.unsqueeze(1), scale_factor=4, mode='bilinear', align_corners=False) 
    #     depth_est_perceptual=depth_est_perceptual.squeeze(1)
    #     #print(depth_est_perceptual.shape)
    #     x_src_perceptual,y_src_perceptual=project_with_depth(depth_est_perceptual, ref_intrinsics_perceptual, ref_extrinsics, src_intrinsics_perceptual, src_extrinsics[i])

    #     #print(x_src_perceptual.shape)

    #     grid=torch.stack((x_src_perceptual.view(batchsize,-1)/((width_perceptual - 1) / 2) - 1,y_src_perceptual.view(batchsize,-1)/((height_perceptual - 1) / 2) - 1),2).unsqueeze(0)
        
    #     sampled_feature_src = F.grid_sample(src_vgg_feature[i][0], grid.view(batchsize, height_perceptual, width_perceptual, 2), mode='bilinear',padding_mode='zeros')

    #         #print(sampled_feature_src.shape) 4*256*128*160 
    #     mask_perpectual=sampled_feature_src>0
    #         #print(ref_vgg_feature[0].shape) 4*256*128*160
    #         #print(len(src_vgg_feature)) 2

    #         #print(len(ref_vgg_feature))

    #     if F.smooth_l1_loss(ref_vgg_feature[0][mask_perpectual],sampled_feature_src[mask_perpectual]).nelement()==0:
    #         loss_perceptual+=torch.tensor(0.)
    #     else:
    #         loss_perceptual+=F.smooth_l1_loss(ref_vgg_feature[0][mask_perpectual],sampled_feature_src[mask_perpectual])


        
        #8
        height_perceptual=int(height*2)
        width_perceptual=int(width*2)
        ref_intrinsics_perceptual=ref_intrinsics
        ref_intrinsics_perceptual[:,:2,:]*=2
        src_intrinsics_perceptual=src_intrinsics[i]
        src_intrinsics_perceptual[:,:2,:]*=2

        depth_est_perceptual=F.interpolate(depth_est.unsqueeze(1), scale_factor=2, mode='bilinear', align_corners=False) 
        depth_est_perceptual=depth_est_perceptual.squeeze(1) 
        #print(depth_est_perceptual.shape)
        x_src_perceptual,y_src_perceptual=project_with_depth(depth_est_perceptual, ref_intrinsics_perceptual, ref_extrinsics, src_intrinsics_perceptual, src_extrinsics[i])

        #print(x_src_perceptual.shape)

        grid=torch.stack((x_src_perceptual.view(batchsize,-1)/((width_perceptual - 1) / 2) - 1,y_src_perceptual.view(batchsize,-1)/((height_perceptual - 1) / 2) - 1),2).unsqueeze(0)
        
        sampled_feature_src = F.grid_sample(src_vgg_feature[i][0], grid.view(batchsize, height_perceptual, width_perceptual, 2), mode='bilinear',padding_mode='zeros')

            #print(sampled_feature_src.shape) 4*256*128*160 
        mask_perpectual=sampled_feature_src>0
            #print(ref_vgg_feature[0].shape) 4*256*128*160 
            #print(len(src_vgg_feature)) 2

            #print(len(ref_vgg_feature))

        if F.smooth_l1_loss(ref_vgg_feature[0][mask_perpectual],sampled_feature_src[mask_perpectual]).nelement()==0:
            loss_perceptual+=torch.tensor(0.)
        else:
            loss_perceptual+=F.smooth_l1_loss(ref_vgg_feature[0][mask_perpectual],sampled_feature_src[mask_perpectual])*0.2


        #15
        grid=torch.stack((x_src.view(batchsize,-1)/((width - 1) / 2) - 1,y_src.view(batchsize,-1)/((height - 1) / 2) - 1),2).unsqueeze(0)

        sampled_feature_src = F.grid_sample(src_vgg_feature[i][1], grid.view(batchsize, height, width, 2), mode='bilinear',padding_mode='zeros')

            #print(sampled_feature_src.shape) 4*256*128*160 
        mask_perpectual=sampled_feature_src>0
            #print(ref_vgg_feature[0].shape) 4*256*128*160
            #print(len(src_vgg_feature)) 2

            #print(len(ref_vgg_feature))

        if F.smooth_l1_loss(ref_vgg_feature[1][mask_perpectual],sampled_feature_src[mask_perpectual]).nelement()==0:
            loss_perceptual+=torch.tensor(0.)
        else:
            loss_perceptual+=F.smooth_l1_loss(ref_vgg_feature[1][mask_perpectual],sampled_feature_src[mask_perpectual])*0.8

        #22
        height_perceptual=int(height/2)
        width_perceptual=int(width/2)
        ref_intrinsics_perceptual=ref_intrinsics
        ref_intrinsics_perceptual[:,:2,:]/=2
        src_intrinsics_perceptual=src_intrinsics[i]
        src_intrinsics_perceptual[:,:2,:]/=2

        depth_est_perceptual=F.interpolate(depth_est.unsqueeze(1), scale_factor=1/2, mode='bilinear', align_corners=False) 
        depth_est_perceptual=depth_est_perceptual.squeeze(1)
        #print(depth_est_perceptual.shape)
        x_src_perceptual,y_src_perceptual=project_with_depth(depth_est_perceptual, ref_intrinsics_perceptual, ref_extrinsics, src_intrinsics_perceptual, src_extrinsics[i])

        #print(x_src_perceptual.shape)

        grid=torch.stack((x_src_perceptual.view(batchsize,-1)/((width_perceptual - 1) / 2) - 1,y_src_perceptual.view(batchsize,-1)/((height_perceptual - 1) / 2) - 1),2).unsqueeze(0)
        
        sampled_feature_src = F.grid_sample(src_vgg_feature[i][2], grid.view(batchsize, height_perceptual, width_perceptual, 2), mode='bilinear',padding_mode='zeros')

            #print(sampled_feature_src.shape) 4*256*128*160 
        mask_perpectual=sampled_feature_src>0
            #print(ref_vgg_feature[0].shape) 4*256*128*160
            #print(len(src_vgg_feature)) 2

            #print(len(ref_vgg_feature))

        if F.smooth_l1_loss(ref_vgg_feature[2][mask_perpectual],sampled_feature_src[mask_perpectual]).nelement()==0:
            loss_perceptual+=torch.tensor(0.)
        else:
            loss_perceptual+=F.smooth_l1_loss(ref_vgg_feature[2][mask_perpectual],sampled_feature_src[mask_perpectual])*0.4


        #29
        # height_perceptual=int(height/4)
        # width_perceptual=int(width/4)
        # ref_intrinsics_perceptual=ref_intrinsics
        # ref_intrinsics_perceptual[:,:2,:]/=4
        # src_intrinsics_perceptual=src_intrinsics[i]
        # src_intrinsics_perceptual[:,:2,:]/=4

        # depth_est_perceptual=F.interpolate(depth_est.unsqueeze(1), scale_factor=1/4, mode='bilinear', align_corners=False) 
        # depth_est_perceptual=depth_est_perceptual.squeeze(1)
        # #print(depth_est_perceptual.shape)
        # x_src_perceptual,y_src_perceptual=project_with_depth(depth_est_perceptual, ref_intrinsics_perceptual, ref_extrinsics, src_intrinsics_perceptual, src_extrinsics[i])

        # #print(x_src_perceptual.shape)

        # grid=torch.stack((x_src_perceptual.view(batchsize,-1)/((width_perceptual - 1) / 2) - 1,y_src_perceptual.view(batchsize,-1)/((height_perceptual - 1) / 2) - 1),2).unsqueeze(0)
        
        # sampled_feature_src = F.grid_sample(src_vgg_feature[i][3], grid.view(batchsize, height_perceptual, width_perceptual, 2), mode='bilinear',padding_mode='zeros')

        #     #print(sampled_feature_src.shape) 4*256*128*160 
        # mask_perpectual=sampled_feature_src>0
        #     #print(ref_vgg_feature[0].shape) 4*256*128*160
        #     #print(len(src_vgg_feature)) 2

        #     #print(len(ref_vgg_feature))

        # if F.smooth_l1_loss(ref_vgg_feature[3][mask_perpectual],sampled_feature_src[mask_perpectual]).nelement()==0:
        #     loss_perceptual+=torch.tensor(0.)
        # else:
        #     loss_perceptual+=F.smooth_l1_loss(ref_vgg_feature[3][mask_perpectual],sampled_feature_src[mask_perpectual])*0.4

        
        #loss_photo 
        #print(src_color.shape) #4*3*128*160
        #print(height) 128
        #print(width) 160
        grid=torch.stack((x_src.view(batchsize,-1)/((width - 1) / 2) - 1,y_src.view(batchsize,-1)/((height - 1) / 2) - 1),2).unsqueeze(0)


        sampled_img_src = F.grid_sample(src_color, grid.view(batchsize, height, width, 2), mode='bilinear', padding_mode='zeros')
        mask=sampled_img_src>0
        #mask=mask*mask_photometric
        #print(torch.sum(mask))
        #print(mask.shape)#1*3*128*160

        #loss=F.smooth_l1_loss(ref_color[mask],sampled_img_src[mask])
        #print("loss_u_shape:{}".format(loss.shape)) size：单
        if F.smooth_l1_loss(ref_color[mask],sampled_img_src[mask]).nelement()==0:
            loss_photo+=torch.tensor(0.)
        else:
            loss_photo+=F.smooth_l1_loss(ref_color[mask],sampled_img_src[mask])

        #计算梯度
        sampled_img_src_dx=gradient_x(sampled_img_src)
        sampled_img_src_dy=gradient_y(sampled_img_src)
        smooth_x=torch.abs(ref_color_dx-sampled_img_src_dx)
        smooth_y=torch.abs(ref_color_dy-sampled_img_src_dy)

        if smooth_x[mask[:,:,:-1,:]].nelement()==0 or smooth_y[mask[:,:,:,:-1]].nelement()==0:
            loss_photo+=torch.tensor(0.)
        else:
            loss_photo+=torch.mean(smooth_x[mask[:,:,:-1,:]])+torch.mean(smooth_y[mask[:,:,:,:-1]])

        if ref_color[mask].nelement()==0:
            loss_ssim+=torch.tensor(0.)
        else:
            loss_ssim+=(1-ssim(ref_color[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0), sampled_img_src[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)))
    #loss_sum=0.0067*loss_s+0.8*loss_photo+0.2*loss_ssim+loss_perceptual
    loss_sum = 0.8 * loss_photo+0.2*loss_ssim+0.0067*loss_s+loss_perceptual
    #print(mask)
    #print(mask.shape)
    return loss_sum,loss_s,loss_photo,loss_ssim,mask,torch.sum(mask),loss_perceptual,loss_normal,normal_by_depth,(depth_est_1-depth_by_normal).abs(),depth_by_normal

# def mvsnet_loss(depth_est, depth_gt, mask):
#     mask = mask > 0.5
#     print(type(depth_est))
#     print(depth_est.shape)
#     print(depth_gt.shape)
#     print(type(mask))
#     print(mask.shape)
#     print(mask)
#     print((depth_est[mask].shape))
#     return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

