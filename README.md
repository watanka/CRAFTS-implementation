
## CRAFTS Reimplementation
---------------------------------------
Character Region Attention for Text Spotting  
Youngmin Baek, Seung Shin, Jeonghun Baek, Sungrae Park, Junyeop Lee, Daehyun Nam, and Hwalsuk Lee  
The full paper is available at: https://arxiv.org/pdf/2007.09629.pdf  


This code was heavily based on these 3 repos below.  
- [craft-reimplementation](https://github.com/backtime92/CRAFT-Reimplementation)
- [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- [ViTSTR](https://github.com/roatienza/deep-text-recognition-benchmark)


main changes in this repo
- changed STD backbone. vgg16 -> ResNet50
- changed STR bacbkone. ResNet50 -> Simplified ResNet
- end2end training from std to str
- orientation feature added, which is well described in the paper
- ViTSTR on str

## TODO
- [ ] random_crop, random_rotate in dataloader.py
- [ ] conditioned data handling by annotation file

