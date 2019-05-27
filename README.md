# Targeted Universal adversarial perturbations

This repos extended [Universal Adversarial Perturbation](https://github.com/LTS4/universal) to target attack.

Only Python3.x ver included.

*python*: Python3.x code to generate universal perturbations using [TensorFlow](https://github.com/tensorflow/tensorflow).Required library is written `requirements.txt` .

## Usage

### Get started

To get started, you can run the demo code to apply a pre-computed universal perturbation for Inception on the image of your choice
```
python demo_inception.py -i data/test_img.png	
```
This will download the pre-trained model, and show the image without and with universal perturbation with the estimated labels.
In this example, the pre-computed targeted universal perturbation in `data/universal.npy` is used. This Perturbation targeted kit fox class.

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same struture as in `demo_inception.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images 
used to compute the perturbation, as well as the feedforward and gradient functions.

## Targeted Perturbations

precomputing_perturbations directory include some precomputing targeted universal adversarial perturbations.

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.


# ターゲット型汎用的摂動(Japanese Docmentation)

このリポジトリは[Universal Adversarial Perturbation](https://github.com/LTS4/universal)をターゲット型攻撃に拡張したものです。Python3.xのコードのみ含みます。

python3.x:[TensorFlow](https://github.com/tensorflow/tensorflow)を使い実装しています。ライブラリ環境は`requirements.txt`に検証時に使用した環境を提供しています

おおよその性質はUniversal Adversarial Perturbaionと同様ですが、摂動を重ね合わせた画像がターゲットへと誤認する割合は60～70%です

## 使い方

### はじめに

まず最初に、あなたの指定した任意の画像について、Inceptionモデルを使い、事前学習によって作られたTargeted Universal Adversarial Perturbationを重ね、識別結果を視覚化します。

```
python demo_inception.py -i data/test_img.png	
```

これを行うと自動的に事前学習されたInceptionモデルをダウンロードし、test_img.pngに「kit fox」をターゲットとしたTargeted Universal Adversarial Perturbationを添加し、識別した結果の文字列とともにを出力します。

![kit fox targeted images](data/index_img/target1.png)

これらの摂動は"Universal"であるため自らが用意した別の画像に対しても重ね合わせるだけで、同様の性質を示すことを確認しています。

100クラスをターゲットとして、あらかじめ摂動を計算しています。ソースコードの上部の`target=1`を（labels.txtを参考にし）1～100の好きに変更することで容易にターゲット摂動を重ね合わせることができます。
例えば`target=3`とすれば次のような結果となります。

![kit fox targeted images](data/index_img/target3.png)

またこれ以外のクラスについての攻撃や自身のモデルにおいて行いたい場合次の章を参照してください。


###  自分のモデルでTargeted Universal Adversarial Perturbationを計算する




## 参考文献

[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.

*Targeted verの論文は卒業論文として執筆したため人にお見せできる論文ではありません。
したがって手法を公開した論文は存在しません。ソースコードからお気持ちを察してください。