# robustness_score
Robustness score: code to reproduce results from "Checking Robustness of Representations Learned by Deep Neural Networks", ECMP 2021 paper 


# Checking Robustness of Representations Learned by Deep Neural Networks [ECML 2021]

Code for our ECML 2021 paper on checking Robustness of DNN. Recent works have shown the vulnerability of deep neural networks to adversarial or out-of-distribution examples. This weakness may come from the fact that training deep models often leads to extracting spurious correlations between image classes and some characteristics of images used for training. As demonstrated, popular, ready-to-use models like the ResNet or the EfficientNet may rely on the non-obvious and counterintuitive features. Detection of these weaknesses is often difficult as classification accuracy is excellent and does not indicate that the model is non-robust. To address this problem, we propose a new method and a measure called robustness score. The method allows indicating which classes are recognized by the deep model using non-robust representations, ie. representations based on spurious correlations. Since the root of this problem lies in the quality of the training data, our method allows us to analyze the training dataset in terms of the existence of these non-obvious spurious correlations. This knowledge can be used to attack the model by finding adversarial images. Consequently, our method can expose threats to the model's reliability, which should be addressed to increase the certainty of classification decisions. The method was verified using the ImageNet and Pascal VOC datasets, revealing many flaws that affect the final quality of deep models trained on these datasets.




## Info

cam.py is based on: [ScoreCAM](https://github.com/yiskw713/ScoreCAM)

## Download resources and set paths

1. Imagenet 2012 


2. Imagenet bboxes

3. Imagenet to wordnet mapping
  -  from (github repo)[https://gist.github.com/symanteve/]  download [imagenet_label_to_wordnet_synset.txt](https://gist.githubusercontent.com/symanteve/a14778f7023dda425e93892217135679/raw/9fa8342b7366596519dfdf53f4db39479eb82de8/imagenet_label_to_wordnet_synset.txt)

Download ImageNet files: images and bboxes

Set paths for images, bboxes and wordnet maping:
PATH_LABEL_TO_WORDNET = '/home/projects/RobutnessScore/imagenet_label_to_wordnet_synset.txt'
PATH_IMAGENET_CLASS = '/home/datasets/imagenet_2012/val/{}/'
PATH_IMAGENET_BBOX = '/home/datasets/imagenet_2012/val/xml/'


## Getting started

Let's start by installing all dependencies. 

`pip install -r requirements.txt`




## Runing

`CUDA_VISIBLE_DEVICES=0 python ImageNet.py`

## Reference

If you find this work helpful, consider citing it. 

```
@inproceedings{szyc2021,
  title={Checking Robustness of Representations Learned by Deep Neural Networks},
  author={Kamil Szyc and Tomasz Walkowiak and Henryk Maciejewski},
 booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
 year={2021}
}
```
