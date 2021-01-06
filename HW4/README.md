# HW4 Image-Super-Resolution

##requirement
tensorflow==1.15

##Training
`python main.py --train --fromscratch --scale <scale> --epochs <epochs> --traindir <path to training dataset> --validdir <path to testing dataset>`

##Inferance
`python inferance.py --scale <scale> --input_dir <path to input folder or image> --output_dir <path to output folder> --ckpt_path <path to ckpt>`

##Reference 

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017.

[2] https://github.com/Saafke/EDSR_Tensorflow


