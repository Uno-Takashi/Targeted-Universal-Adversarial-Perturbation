# Universal adversarial perturbations

*matlab*: MATLAB code to generate universal perturbations, using [Caffe](https://github.com/BVLC/caffe) or [MatConvNet](https://github.com/vlfeat/matconvnet).

*python*: Python code to generate universal perturbations using [TensorFlow](https://github.com/tensorflow/tensorflow).

*precomputed*: Precomputed universal perturbations, for pre-trained models on ImageNet.

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017


# Python

Python code to find a universal perturbation [[1]](http://arxiv.org/pdf/1610.08401), using the [TensorFlow](https://www.tensorflow.org/) library.

## Usage

### Get started

To get started, you can run the demo code to apply a pre-computed universal perturbation for Inception on the image of your choice
```
python demo_inception.py -i data/test_img.png	
```
This will download the pre-trained model, and show the image without and with universal perturbation with the estimated labels.
In this example, the pre-computed universal perturbation in `data/universal.npy` is used.

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same struture as in `demo_inception.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images 
used to compute the perturbation, as well as the feedforward and gradient functions.

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
