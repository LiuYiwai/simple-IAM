# simple-IAM
A simple PyTorch implementation of Learning Instance Activation Maps for Weakly Supervised Instance Segmentation, in CVPR 2019 (Spotlight)



A simple implementation as my homework, modified based on [ultra-thin-PRM](https://github.com/chuchienshu/ultra-thin-PRM).

Implementation details With my own understanding of the paper, it may be different from the author.

This implementation is better when training demos and other data sets with fewer samples, but when I use VOC2012 training, it is not ideal.  :(

If you have any good suggestions, please let me know. Thank you !



## Update

Mar. 24. 2020:

​	Fixed a bug, this bug will cause the filling module to only affect the first PRM.

​	Updated the transform implementation of proposals, data augmentation can now be used in the training filling module.

​	



### Reference
```markdown
@article{Zhu2019IAM,
    title={{Learning Instance Activation Maps for Weakly Supervised Instance Segmentation}},
    author = {Zhu, Y. and Zhou, Y. and Xu, H. and Ye, Q. and Doermann, D. and Jiao, J.},
    booktitle = {CVPR},
    year = {2019}
}
```
