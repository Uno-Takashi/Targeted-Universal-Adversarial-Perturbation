# Targeted Universal adversarial perturbations

This repos extended [Universal Adversarial Perturbation](https://github.com/LTS4/universal) to target attack.

Only Python3.x ver included.

*python*: Python3.x code to generate universal perturbations using [TensorFlow](https://github.com/tensorflow/tensorflow).


# Python

Python code to find a universal perturbation [[1]](http://arxiv.org/pdf/1610.08401), using the [TensorFlow](https://www.tensorflow.org/) library.

## Usage

### Get started

To get started, you can run the demo code to apply a pre-computed universal perturbation for Inception on the image of your choice
```
python demo_inception.py -i data/test_img.png	
```
This will download the pre-trained model, and show the image without and with universal perturbation with the estimated labels.
In this example, the pre-computed targeted universal perturbation in `data/universal.npy` is used. This Perturbation tageted kit fox class.

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same struture as in `demo_inception.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images 
used to compute the perturbation, as well as the feedforward and gradient functions.

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.


# ターゲット型汎用的摂動(Japanese Docmentation)

このリポジトリは[Universal Adversarial Perturbation](https://github.com/LTS4/universal)をターゲット型攻撃に拡張したものです。Python3.xのコードのみ含みます。

python3.x:[TensorFlow](https://github.com/tensorflow/tensorflow)を使い実装しています。

## 使い方

### はじめに

まず最初に、あなたの指定した任意の画像について、Inceptionモデルを使い、事前学習によって作られたTargeted Universal Adversarial Perturbationを重ね、識別結果を視覚化します。

```
python demo_inception.py -i data/test_img.png	
```

これを行うと自動的に事前学習されたInceptionモデルをダウンロードし、test_img.pngにTargeted Universal Adversarial Perturbationを添加し、識別した結果の文字列とともにを出力します。

###  自分のモデルでTargeted Universal Adversarial Perturbationを計算する