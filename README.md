# Neural Style Transfer
Draw an image in the style of another one as presented in the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). 


## Development Environment
1、Conda 4.13.0

2、Python 3.9.12

3、Tensorflow 2.9.1

## How to use the code
```shell
python main.py [content path] [style path]
```

For example transfer the style of vangogh into the monalisa painting:
```shell
python main.py monalisa.png vangogh.png
```

## Source
Adapted from [the tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
