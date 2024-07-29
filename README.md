### ComfyUI_AnyDoor
You can using  AnyDoor in comfyUI，you can change the clothes of the characters and move the position of the objects in the picture

AnyDoor origin From: [AnyDoor](https://github.com/ali-vilab/AnyDoor) 
----

Tips：  
At present, there is no good adaptation node for the injection of object mask positions, so we will add one later.     
If you have pre downloaded the model, you don't need to install the  modelscope library   

英文阅读慢的，请看[中文说明](https://github.com/smthemex/ComfyUI_AnyDoor/blob/main/README-CN.md)

My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   

1.Installation
---
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_AnyDoor.git
```
or  using ComfyUI-Manager 

2.requirements  
---
```
pip install -r requirements.txt
```
if miss any modules,rename "for_modules_miss_requirements.txt" to  requirements.txt    
and pip install -r requirements.txt     
or just "pip install missing module"   

3  Required models
---
*model 1：*   

<online>    
The first use will automatically download the model, but the "Pruned" model is not on the huggingface server, so there is a "modelscope" (魔搭)library in the requirements, which will reference the use of the modelscope library to download the corresponding "Pruned" model    

<offline>   

--using "origin" model  download "epoch=1-step=8687.ckpt"(16.8G) : [link](https://huggingface.co/spaces/xichenhku/AnyDoor/tree/main)      
or    
--using "pruned" model  download "epoch=1-step=8687-pruned.ckpt"(4.9G) : [link](https://modelscope.cn/models/bdsqlsz/AnyDoor-Pruned/files)    

*model 2：*   

<online>   
The model is not stored on the huggingface, you can find it at the following two URLs.The first use will automatically download the model from “modelscope”    

<offline>    
  
--download "dinov2_vitg14_pretrain.pth"(4.5G) :[link](https://modelscope.cn/models/bdsqlsz/AnyDoor-Pruned/files)   
or   
--download "dinov2_vitg14_pretrain.pth"(4.5G) :[link](https://github.com/facebookresearch/dinov2?tab=readme-ov-file)      

--The file storage address is as follows:  

```
├── ComfyUI/models/
|      ├──anydoor/
|             ├── dinov2_vitg14_pretrain.pth
|             ├── epoch=1-step=8687.ckpt  (origin)
|             ├── epoch=1-step=8687-pruned.ckpt  (pruned)
```
4 Using
---

4.1 Image and image_mask are the objects you want to transfer, while bg_image and bg_mask are the background of the image, which is the main image;   
4.2 Seed is currently invalid;   
Although the built-in mask of comfyUI can also be used, I still recommend using Seg or Sam;   
4.4 If there is not enough graphics memory, you can consider enabling save_memory;    
4.5 useInteractive_seg will process the mask again, unless you are a hand animation mask, it is generally not recommended to open it;   
4.6 The maximum model training size is 768 * 768, so it is not recommended to use oversized images for image training;    
4.7 The workflow file is located in the examples folder;   
4.8 When inputting the mask image, please make sure that the selection you want to replace is whiter.    


5 Example
---
change cloth using SAM...   
![](https://github.com/smthemex/ComfyUI_AnyDoor/blob/main/examples/example.png)

change cloth using normal mask... 
![](https://github.com/smthemex/ComfyUI_AnyDoor/blob/main/examples/img2img.png)


Citation
---
特别致谢： [青龍聖者@bdsqlsz](https://github.com/sdbds) 的“poch=1-step=8687-pruned.ckpt” 模型

AnyDoor
``` python  
@article{chen2023anydoor,
  title={Anydoor: Zero-shot object-level image customization},
  author={Chen, Xi and Huang, Lianghua and Liu, Yu and Shen, Yujun and Zhao, Deli and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2307.09481},
  year={2023}
}
```

Citing DINOv2
```python
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```python
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
