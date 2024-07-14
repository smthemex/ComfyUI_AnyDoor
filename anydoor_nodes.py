# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import hashlib
import os
import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from .ldmx.model import create_model, load_state_dict
from .ldmx.ddim_hacked import DDIMSampler
from .ldmx.hack import disable_verbosity, enable_sliced_attention
from .datasets.data_utils import *
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import folder_paths
from modelscope.hub.file_download import model_file_download


anydoor_current_path = os.path.dirname(os.path.abspath(__file__))
weigths_current_path = os.path.join(folder_paths.models_dir, "anydoor")

if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)
    
    
if "anydoor" not in folder_paths.folder_names_and_paths:
    node_current_paths = [os.path.join(folder_paths.models_dir, "anydoor")]
else:
    node_current_paths, _ = folder_paths.folder_names_and_paths["anydoor"]
    print(node_current_paths)
folder_paths.folder_names_and_paths["anydoor"] = (node_current_paths, folder_paths.supported_pt_extensions)


# Tensor to PIL  NCHW 2 CV
def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

# torch Tensor to CV2  NCHW 2 CV
def tensor2cv(tensor):
    tensor=tensor.to(torch.device("cpu"))
    tensor=tensor.squeeze() #去掉批次
    tensor=tensor.mul_(255).add_(0.5).clamp_(0,255).permute(1,2,0).type(torch.uint8).numpy()
    cv_img =cv2.cvtColor(tensor,cv2.COLOR_BGR2RGB)
    return cv_img

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv(img):
    ar_img=np.array(img)
    cv_img=cv2.cvtColor(ar_img, cv2.COLOR_RGB2BGR)
    return cv_img
    
# PIL Hex
def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

# Mask to PIL
def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil

# Tensor to SAM-compatible NumPy
def tensor2sam(image):
    # Convert tensor to numpy array in HWC uint8 format with pixel values in [0, 255]
    sam_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    # Transpose the image to HWC format if it's in CHW format
    if sam_image.shape[0] == 3:
        sam_image = np.transpose(sam_image, (1, 2, 0))
    return sam_image

# SAM-compatible NumPy to tensor
def sam2tensor(image):
    # Convert the image to float32 and normalize the pixel values to [0, 1]
    float_image = image.astype(np.float32) / 255.0
    # Transpose the image from HWC format to CHW format
    chw_image = np.transpose(float_image, (2, 0, 1))
    # Convert the numpy array to a tensor
    tensor_image = torch.from_numpy(chw_image)
    return tensor_image

def cv2narry(img):
    # img=cv2.ctvtColor(img,cv2.COLOR_BGR2RGB)
    # nump_img=np.array(img)
    nump_img=torch.FloatTensor(img).permute(2,0,1)
    return nump_img

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

# def proces_img(img):
#     image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
#     mask = (image[:, :, -1] > 128).astype(np.uint8)
#     image = image[:, :, :-1]
#     image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
#     return  image,mask

def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image)

    # collage aug
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask)
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) )
    return item

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(save_memory,model,ddim_sampler,ref_image, ref_mask, tar_image, tar_mask, guidance_scale,seed,steps,strength,width,height,batch_size):
    if save_memory=="ture":
        save_memory=True
    else:
        save_memory=False

    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)

    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    #seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg']
    hint = item['hint']
    #num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control for _ in range(batch_size)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda()
    clip_input = torch.stack([clip_input for _ in range(batch_size)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    #H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * batch_size)]}
    shape = (4, height // 8, width // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    #num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    #image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    #strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    #ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    #scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(steps, batch_size,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=guidance_scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
    return gen_image


class AnyDoor_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_memory": ("BOOLEAN", {"default": False},),
                "ckpts": (["pruned", "origin",],),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL","STRING",)
    RETURN_NAMES = ("model","ddim_sampler","info",)
    FUNCTION = "main_loader"
    CATEGORY = "AnyDoor"

    def main_loader(self,save_memory,ckpts):
        disable_verbosity()
        if save_memory:
            enable_sliced_attention()
            
        # download model
        model_config = os.path.join(anydoor_current_path, "configs", "anydoor.yaml")
        dino_model_path = os.path.join(weigths_current_path, "dinov2_vitg14_pretrain.pth")
        if not os.path.exists(dino_model_path):
            model_file_download('bdsqlsz/AnyDoor-Pruned', file_path="dinov2_vitg14_pretrain.pth",
                                local_dir=weigths_current_path)
        if ckpts == "pruned":
            model_ckpt = os.path.join(weigths_current_path, "epoch=1-step=8687-pruned.ckpt")
            if not os.path.exists(model_ckpt):
                model_ckpt = model_file_download('bdsqlsz/AnyDoor-Pruned',
                                                 file_path="epoch=1-step=8687-pruned.ckpt",
                                                 local_dir=weigths_current_path)
        else:
            model_ckpt = os.path.join(weigths_current_path, "epoch=1-step=8687.ckpt")
            if not os.path.exists(model_ckpt):
                model_ckpt = model_file_download('iic/AnyDoor', file_path="epoch=1-step=8687.ckpt",
                                                 local_dir=weigths_current_path)
        model = create_model(model_config).cpu()
        model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)
        if save_memory:
            info = "true"
        else:
            info = "false"

        return (model,ddim_sampler,info)

class AnyDoor_img2img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_mask": ("IMAGE",),
                "bg_image": ("IMAGE",),
                "bg_mask": ("IMAGE",),
                "model": ("MODEL",),
                "ddim_sampler": ("MODEL",),
                "info": ("STRING", {"forceInput": True}),
                "cfg": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, }),
                "width": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1, "display": "number"}),
                "use_interactive_seg":("BOOLEAN", {"default": False},),}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "AnyDoor"
    FUNCTION = "anydoor_main"
    
    def anydoor_main(self,image,image_mask,bg_image,bg_mask,model,ddim_sampler,info,cfg,seed,steps,control_strength,width,height,batch_size,use_interactive_seg):
        
        bg_image = tensor2pil(bg_image)
        bg_image = np.asarray(bg_image)
        
        bg_mask=tensor2pil(bg_mask).convert("L")
        bg_mask = np.asarray(bg_mask)
        bg_mask = np.where(bg_mask > 128, 1, 0).astype(np.uint8)
        
        ref_image = tensor2pil(image)
        ref_image = np.asarray(ref_image)
        
        ref_mask =tensor2pil(image_mask).convert("L")
        ref_mask = np.asarray(ref_mask)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)
        
        if use_interactive_seg:
            from .iseg.coarse_mask_refine_util import BaselineModel
            model_path = os.path.join(anydoor_current_path, "iseg","coarse_mask_refine.pth")
            iseg_model = BaselineModel().eval()
            weights = torch.load(model_path, map_location='cpu')['state_dict']
            iseg_model.load_state_dict(weights, strict=True)
            img = torch.from_numpy(ref_image.transpose((2, 0, 1)))
            img = img.float().div(255).unsqueeze(0)
            mask = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0)
            pred = iseg_model(img, mask)['instances'][0, 0].detach().numpy() > 0.5
            ref_mask=pred.astype(np.uint8)

        gen_image = inference_single_image(info,model,ddim_sampler,ref_image, ref_mask, bg_image.copy(), bg_mask,cfg,seed,steps,control_strength,width,height,batch_size)
        image=phi2narry(gen_image)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "AnyDoor_LoadModel":AnyDoor_LoadModel,
    "AnyDoor_img2img": AnyDoor_img2img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyDoor_LoadModel":"AnyDoor_LoadModel",
    "AnyDoor_img2img": "AnyDoor_img2img",
}
