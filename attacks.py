# Object-based diverse input method
# Anonymous CVPR submission

import torch.nn as nn

import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import math
import os
import sys
import numpy as np
import sys
import cv2
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

from PIL import Image
from config import *


## Pytorch3D ########################################
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    look_at_rotation,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    blending
)

##########################################


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def plot_img(img_tensor, file_name):
    img = np.array(img_tensor[0].cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    img = img.astype(np.uint8)

    im = Image.fromarray(img)
    im.save("imgs/" + file_name + ".png")

class Render3D(object):
    def __init__(self,config_idx=1,count=1):

        exp_settings=exp_configuration[config_idx] # Load experiment configuration

        self.config_idx=config_idx
        self.count=count
        self.eval_count=0


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        raster_settings = RasterizationSettings(
            image_size=299, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Just initialization. light position and brightness are randomly set for each inference 
        self.lights = PointLights(device=self.device, ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.5, 0.5, 0.5), ), specular_color=((0.5, 0.5, 0.5), ), 
        location=[[0.0, 3.0,0.0]])

        R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        self.materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=exp_settings['shininess']
        )

        # Note: the background color of rendered images is set to -1 for proper blending
        blend_params = blending.BlendParams(background_color=[-1., -1., -1.])


        # Create a renderer by composing a mesh rasterizer and a shader. 
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                blend_params=blend_params
            )
        )
        # 3D Model setting
        # {'3d model name', ['filename', x, y, w, h, initial distance, initial elevation, initial azimuth, initial translation]}
        self.model_settings={'pack':['pack.obj',255,255,510,510,1.2,0,0,[0,0.02,0.]],
        'cup':['cup.obj',693,108,260,260,1.7,0,0,[0.,-0.1,0.]],
        'pillow':['pillow.obj',10,10,470,470,1.7,0,0],
        't_shirt':['t_shirt_lowpoly.obj',180,194,240,240,1.2,0,0,[0.0,0.05,0]],
        'book':['book.obj',715,66,510,510,1.3,0,0,[0.3,0.,0]],
        '1ball':['1ball.obj',359,84,328,328,2.1,-40,-10],
        '2ball':['2ball.obj',359,84,328,328,1.9,-40,-10,[-0.1,0.,0]],
        '3ball':['3ball.obj',359,84,328,328,1.8,-25,-10,[-0.1,0.15,0]],
        '4ball':['4ball.obj',359,84,328,328,1.8,-25,-10,[0.,0.1,0]]
        }


        self.source_models=exp_settings['source_3d_models'] # Import source model list

        self.background_img=torch.zeros((1,3,299,299)).to(device)
        
        for src_model in self.source_models:
                self.model_settings[src_model][0]=load_object(self.model_settings[src_model][0])

        # The following code snippet is for 'blurred image' backgrounds.
        kernel_size=50
        kernel = gkern(kernel_size, 15).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        self.gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

    def render(self, img):
        self.eval_count+=1
        
        exp_settings=exp_configuration[self.config_idx]

        # Default experimental settings.
        if 'background_type' not in exp_settings:
            exp_settings['background_type']='none'
        if 'texture_type' not in exp_settings:
            exp_settings['texture_type']='none'
        if 'visualize' not in exp_settings:
            exp_settings['visualize']=False

        x_adv=img
        # Randomly select an object from the source object pool
        pick_idx=np.random.randint(low=0,high=len(self.source_models))

        # Load the 3D mesh
        mesh=self.model_settings[self.source_models[pick_idx]][0]

        # Load the texture map
        texture_image=mesh.textures.maps_padded()

        texture_type=exp_settings['texture_type']

        if texture_type=='random_pixel':
            texture_image.data=torch.rand_like(texture_image,device=device)
        elif texture_type=='random_solid': # Default setting
            texture_image.data=torch.ones_like(texture_image,device=device)*(torch.rand((1,1,1,3),device=device)*0.6+0.1)
        elif  texture_type=='custom':
            texture_image.data=torch.ones_like(texture_image,device=device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,1,1,3)).to(device)
        
        (pattern_h,pattern_w)=(self.model_settings[self.source_models[pick_idx]][4],self.model_settings[self.source_models[pick_idx]][3])

        # Resize the input image
        resized_x_adv=F.interpolate(x_adv, size=(pattern_h, pattern_w), mode='bilinear').permute(0,2,3,1)
        # Insert the resized image into the canvas area of the texture map
        (x,y)=self.model_settings[self.source_models[pick_idx]][1],self.model_settings[self.source_models[pick_idx]][2]
        texture_image[:,y:y+pattern_h,x:x+pattern_w,:]=resized_x_adv

        # Adjust the light parameters
        self.lights.location = torch.tensor(exp_settings['light_location'], device=device)[None]+(torch.rand((3,), device=device)*exp_settings['rand_light_location']-exp_settings['rand_light_location']/2)
        self.lights.ambient_color=torch.tensor([exp_settings['ambient_color']]*3, device=device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_ambient_color'])
        self.lights.diffuse_color=torch.tensor([exp_settings['diffuse_color']]*3, device=device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_diffuse_color'])
        self.lights.specular_color=torch.tensor([exp_settings['specular_color']]*3, device=device)[None]

        
        # Adjust the camera parameters
        rand_elev=torch.randint(exp_settings['rand_elev'][0],exp_settings['rand_elev'][1]+1, (1,))
        rand_azim=torch.randint(exp_settings['rand_azim'][0],exp_settings['rand_azim'][1]+1, (1,))
        rand_dist=(torch.rand((1,))*exp_settings['rand_dist']+exp_settings['min_dist'])
        rand_angle=torch.randint(exp_settings['rand_angle'][0],exp_settings['rand_angle'][1]+1, (1,))



        R, T = look_at_view_transform(dist=(self.model_settings[self.source_models[pick_idx]][5])*rand_dist, elev=self.model_settings[self.source_models[pick_idx]][6]+rand_elev, 
        azim=self.model_settings[self.source_models[pick_idx]][7]+rand_azim,up=((0,1,0),))

        if len(self.model_settings[self.source_models[pick_idx]])>8: # Apply initial translation if it is given.
            TT=T+torch.FloatTensor(self.model_settings[self.source_models[pick_idx]][8])
        else:
            TT=T

        # Compute rotation matrix for tilt
        angles=torch.FloatTensor([[0,0,rand_angle*math.pi/180]]).to(device)
        rot=compute_rotation(angles).squeeze()
        R=R.to(device)

        R=torch.matmul(rot,R)

        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=TT)

        # Render the mesh with the modified rendering environments.
        rendered_img = self.renderer(mesh, lights=self.lights, materials=self.materials, cameras=self.cameras)

        rendered_img=rendered_img[:, :, :,:3] # RGBA -> RGB

        rendered_img=rendered_img.permute(0,3,1,2) # B X H X W X C -> B X C X H X W

        background_type=exp_settings['background_type']
        
        # The following code snippet is for blending
        rendered_img_mask = 1.-(rendered_img.sum(dim=1,keepdim=True)==-3.).float()
        rendered_img = torch.clamp(rendered_img, 0., 1.)
        if background_type=='random_pixel':
            background_img=torch.rand_like(rendered_img,device=device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='random_solid':
            background_img=torch.ones_like(rendered_img,device=device)*torch.rand((1,3,1,1),device=device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='blurred_image':
            background_img=img.clone().detach()
            background_img = F.conv2d(background_img, self.gaussian_kernel, bias=None, stride=1, padding='same', groups=3)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='custom':
            background_img=torch.ones_like(rendered_img,device=device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,3,1,1)).to(device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        else:
            result_img=rendered_img

        if exp_settings['visualize']==True:
            result_img_npy=result_img.permute(0,2,3,1)
            result_img_npy=result_img_npy.squeeze().cpu().detach().numpy()
            converted_img=cv2.cvtColor(result_img_npy, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video', converted_img) #[0, ..., :3]
            key=cv2.waitKey(1) & 0xFF

        return result_img
def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)

def rigid_transform( vs, rot, trans):
    vs_r = torch.matmul(vs, rot)
    vs_t = vs_r + trans.view(-1, 1, 3)
    return vs_t

def load_object(obj_file_name):
    obj_filename = os.path.join("./data", obj_file_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the 3D model using load_obj
    verts, faces, aux = load_obj(obj_filename)
    
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the mesh. 
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    angles=torch.FloatTensor([[90*math.pi/180,0,0]]).to(device)

    rot=compute_rotation(angles).squeeze()

    verts=torch.matmul(verts,rot)

    # Get the scale normalized textured mesh
    mesh = load_objs_as_meshes([obj_filename], device=device)
    mesh = Meshes(verts=[verts], faces=[faces_idx],textures=mesh.textures)


    return mesh

def render_3d_aug_input(x_adv, renderer,prob=0.7):
    c = np.random.rand(1)
    if c <= prob:
        x_ri=x_adv.clone()
        for i in range(x_adv.shape[0]):
            x_ri[i]=renderer.render(x_adv[i].unsqueeze(0))
        return  x_ri 
    else:
        return  x_adv



def calculate_v(model, x_adv_or_nes, y, eps, number_of_v_samples, beta, target_label, attack_type, number_of_si_scales, prob,loss_fn,renderer):
    sum_grad_x_i = torch.zeros_like(x_adv_or_nes)
    for i in range(number_of_v_samples):
        x_i = x_adv_or_nes.clone().detach() + (torch.rand(x_adv_or_nes.size()).cuda()*2-1.) * (beta * eps)
        x_i.requires_grad = True
        if 'S' in attack_type: 
            ghat = calculate_si_ghat(model, x_i, y, number_of_si_scales, target_label, attack_type, prob,loss_fn,renderer)
        else:
            if 'D' in attack_type:
                x_i2 = DI(x_i,prob)
            elif 'R' in attack_type:
                x_i2 = RDI(x_i)
            elif 'O' in attack_type:
                x_i2 = render_3d_aug_input(x_i,renderer=renderer,prob=prob)
            else:
                x_i2 = x_i
            output_x_adv_or_nes = model(x_i2)
            loss= loss_fn(output_x_adv_or_nes)
            ghat = torch.autograd.grad(loss, x_i,
                    retain_graph=False, create_graph=False)[0]
        sum_grad_x_i += ghat.detach()
    v = sum_grad_x_i / number_of_v_samples
    return v


def calculate_si_ghat(model, x_adv_or_nes, y, number_of_si_scales, target_label, attack_type, prob, loss_fn,renderer):
    x_neighbor = x_adv_or_nes.clone().detach()
    grad_sum = torch.zeros_like(x_neighbor).cuda()
    for si_counter in range(0, number_of_si_scales):
        si_div = 2 ** si_counter
        si_input = (((x_adv_or_nes.clone().detach()-0.5)*2 / si_div)+1)/2 # 0 1 -> -1 1
        si_input.requires_grad = True
        # Diverse-Input
        if 'D' in attack_type:
            si_input2 = DI(si_input,prob)
        elif 'R' in attack_type:
            si_input2 = RDI(si_input)
        elif 'O' in attack_type:
            si_input2 = render_3d_aug_input(si_input,renderer=renderer,prob=prob)
        else:
            si_input2 = si_input
        output_si = model(si_input2)

        loss_si=loss_fn(output_si)
        si_input_grad = torch.autograd.grad(loss_si, si_input,
                retain_graph=False, create_graph=False)[0]
        grad_sum += si_input_grad*(1/si_div)

    ghat = grad_sum
    return ghat


def DI(X_in,prob):
    rnd = np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    c = np.random.rand(1)
    if c <= prob:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in
    

def RDI(x_adv):
    x_di = x_adv 
    di_pad_amount=340-299
    
    di_pad_value=0

    ori_size = x_di.shape[-1]
    rnd = int(torch.rand(1) * di_pad_amount) + ori_size
    x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1) * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1) * pad_max)
    pad_bottom = pad_max - pad_top
    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
    x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)


    return x_di


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel



class CELoss(nn.Module):
    def __init__(self, labels):
        super(CELoss, self).__init__()
        self.labels=labels
        self.ce=nn.CrossEntropyLoss(reduction='mean')
        self.labels.requires_grad = False
    def forward(self, logits):
        return -self.ce(logits, self.labels)



class LogitLoss(nn.Module):
    def __init__(self, labels):
        super(LogitLoss, self).__init__()
        self.labels=labels
        self.labels.requires_grad = False

    def forward(self, logits):
        real = logits.gather(1,self.labels.unsqueeze(1)).squeeze(1)
        logit_dists = ( -1 * real)
        loss = logit_dists.sum()
        return -loss

class ATTA(nn.Module):
    def __init__(self):
        super(ATTA, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 16, padding="same", groups=1,bias=False)
        self.lr=nn.LeakyReLU(0.2)
        self.conv2=nn.Conv2d(3, 3, 3, padding="same",groups=1, bias=False)

        # The details of the weight initialization method are not described in the ATTA paper.
        # In order to train the network in 10 iterations, a small random noise is added to the dirac initialized weights.

        torch.nn.init.dirac_(self.conv1.weight, 1)
        torch.nn.init.dirac_(self.conv2.weight, 1)
        self.conv1.weight.data+=torch.randn_like(self.conv1.weight.data)*0.01
        self.conv2.weight.data+=torch.randn_like(self.conv2.weight.data)*0.01


    def forward(self, x):
        x2 = self.conv1(x)
        x3 = self.lr(x2)
        x4 = self.conv2(x3)
        return x4

def ATTA_aug_input(x_adv, atta_models):
    x_ri=x_adv.clone()
    for i in range(x_adv.shape[0]): # Batch processing
        x_ri[i]=atta_models[i](x_adv[i].unsqueeze(0))
    x_ri=x_ri.clamp(0,1)
    return  x_ri







def advanced_fgsm(attack_type, model, x, y, target_label=None, num_iter=10, max_epsilon=16, mu=1.0, number_of_v_samples=5, beta=1.5,
                     number_of_si_scales=5, count=0, config_idx=1):
    """Perform advanced fgsm attack

    Args:
        attack_type: string containing 'M'(momentum) or 'N'(Nesterov momentum) /
        'D' (Diverse input) or 'R' (Resized-diverse-input) or 'O' (Object-based diverse input) /
        'V'(variance tuning) / 'S'(Scale invariance) / 'T' (Translation-invariance)
        model: the target model
        x: a batch of images.
        y: true labels corresponding to the batch of images
        target_label : used for targeted attack. 
        num_iter: T. number of iterations to perform.
        max_epsilon: Linf norm of resulting perturbation (in pixels)
        mu: mu. decay of momentum.
        number_of_v_samples: N. # samples to calculate V
        beta: the bound for variance tuning.
        number_of_si_scales: m. (in scale-invariance paper)
    Returns:
        The batch of adversarial examples corresponding to the original images
    """


    
    exp_settings=exp_configuration[config_idx]
    renderer=Render3D(config_idx=config_idx,count=count)
    prob=exp_settings['p']
    lr=exp_settings['alpha'] # Step size alpha
    number_of_si_scales=exp_settings['number_of_si_scales']
    number_of_v_samples=exp_settings['number_of_v_samples']
    if 'save_img' not in exp_settings:
        exp_settings['save_img']=False

    if "M" not in attack_type and "N" not in attack_type:
        mu = 0

    


    ti_kernel_size=5

    if 'T' in attack_type:
        kernel = gkern(ti_kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

    model.eval()

    eps = max_epsilon / 255.0  # epsilon in scale [0, 1]
    alpha = lr / 255.0

    x_min = torch.clamp(x - eps, 0.0, 1.0)
    x_max = torch.clamp(x + eps, 0.0, 1.0)

    x_adv = x.clone()

    if 'A' in attack_type: # ATTA
        mse=torch.nn.MSELoss(reduction='sum')

        # We use the default hyper parameters of ATTA
        K_outer=10
        K_inner=10

        alpha_1=1.0
        alpha_2=10.0
        atta_beta=1.0
        ATTA_models=[]
        ce=nn.CrossEntropyLoss(reduction='sum')
        logit_loss_fn_target=LogitLoss(target_label)

        for i in range(x.size()[0]):
            atta_model=ATTA().cuda() # Randomly initialize theta
            model_optimizer = torch.optim.Adam(atta_model.parameters(), lr=0.001)
            temp_x_adv=x_adv.clone().detach()
            temp_x_adv.requires_grad=True
            adv_optimizer = torch.optim.Adam([temp_x_adv], lr=0.01)
            for ko in range(K_outer):
                atta_model.eval()
                for ki in range(K_inner):
                    # Update x_adv
                    temp_x_adv2=temp_x_adv
                    x_at=atta_model(temp_x_adv2)
                    #L_fool=-ce(model(x_at),y)-atta_beta*ce(model(temp_x_adv2),y)

                    # Instead of CE loss for non-targeted attack, we use the simple logit loss for generating targeted adversarial examples
                    # We observe this improves the targeted atttack success rates.
                    L_fool=logit_loss_fn_target(model(x_at))+atta_beta*logit_loss_fn_target(model(temp_x_adv2))
                    
                    adv_optimizer.zero_grad()
                    L_fool.backward()
                    adv_optimizer.step()
                    temp_x_adv.data = torch.clamp(temp_x_adv.data, x_min, x_max)
                    
                model_optimizer.zero_grad()
                atta_model.train()
                L_T=ce(model(atta_model(temp_x_adv)),y)+alpha_1*ce(model(atta_model(x)),y)+alpha_2*mse(atta_model(temp_x_adv),temp_x_adv) #alpha_2*torch.norm(atta_model(temp_x_adv)-temp_x_adv,p=2)**2 #
                #L_T=logit_loss_fn(model(atta_model(temp_x_adv)))+alpha_1*logit_loss_fn(model(atta_model(x)))+alpha_2*mse(atta_model(temp_x_adv),temp_x_adv)

                model_optimizer.zero_grad()
                L_T.backward()
                model_optimizer.step()
            atta_model.eval()
            ATTA_models.append(atta_model)

    g = 0
    v = 0

    if '3' in attack_type:
        loss_fn=LogitLoss(target_label)
    else:
        loss_fn=CELoss(target_label)

    B,C,H,W=x_adv.size()
    x_advs=torch.zeros((num_iter//20,B,C,H,W)).to(device)

    for t in range(num_iter):
        # Calculate ghat
        if 'N' in attack_type:  # Nesterov momentum
            x_nes = x_adv.detach() + alpha * mu * g  # x_nes = x + alpha * momentum * grad
        else:  # usual momentum
            x_nes = x_adv.detach()
        x_nes.requires_grad = True
        if 'S' in attack_type:  # Scale-Invariance
            ghat = calculate_si_ghat(model, x_nes, y, number_of_si_scales, target_label, attack_type,
                                        prob, loss_fn,renderer)
        else:
            if exp_settings['save_img'] and count in exp_settings['target_img_idx']:
                plot_img(x_nes,str(count)+'_'+str(t)+'_adv')

            if 'D' in attack_type:
                x_adv_or_nes = DI(x_nes,prob) 
            elif 'R' in attack_type:
                x_adv_or_nes = RDI(x_nes)
            elif 'O' in attack_type:
                x_adv_or_nes = render_3d_aug_input(x_nes,renderer=renderer,prob=prob)
            else:
                x_adv_or_nes = x_nes
            
            if exp_settings['save_img'] and count in exp_settings['target_img_idx']:
                plot_img(x_adv_or_nes,str(count)+'_'+str(t)+'_transformed')

            output2 = model(x_adv_or_nes)
            loss = loss_fn(output2)

            if 'A' in attack_type:
                x_adv_or_nes2 = ATTA_aug_input(x_adv_or_nes,ATTA_models)
                output3 = model(x_adv_or_nes2)
                loss += loss_fn(output3)

            ghat = torch.autograd.grad(loss, x_nes,
                                        retain_graph=False, create_graph=False)[0]

        # Update g
        grad_plus_v = ghat + v  

        if 'T' in attack_type:  # Translation-invariance
            grad_plus_v = F.conv2d(grad_plus_v, gaussian_kernel, bias=None, stride=1, padding=((ti_kernel_size-1)//2,(ti_kernel_size-1)//2), groups=3) #TI

        if 'M' in attack_type or 'N' in attack_type:
            g = mu * g + grad_plus_v / torch.sum(torch.abs(grad_plus_v),dim=[1,2,3],keepdim=True)
        else:
            g=grad_plus_v

        # Update v
        if 'V' in attack_type:
            v = calculate_v(model, x_nes, y, eps, number_of_v_samples, beta, target_label, attack_type,
                            number_of_si_scales, prob,loss_fn,renderer) - ghat

        # Update x_adv
        pert = alpha * g.sign()
        x_adv = x_adv.detach() + pert
        x_adv = torch.clamp(x_adv, x_min, x_max)
        if (t+1)%20==0:
            x_advs[(t+1)//20-1]=x_adv.clone().detach()

    torch.cuda.empty_cache() 

    return x_advs.detach()



