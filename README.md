# tianshu-diffusion
An homage to Tian Shu (天书) by Xu Bing.

## Usage

Install the dependencies with
```
py -m pip install -r requirements.txt
```

First the image data needs to be generated, this is done using
```
mkdir images
py data_generator.py
```
Then the images directory will be populated with 32x32 images of every standard
Chinese character.

Training is then done using
```py train.py```
