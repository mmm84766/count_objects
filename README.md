## How to Run

### Install required libraries
`pip3 install -r requirements.txt`


### Step 1: Annotate some images
- Save some photos with your custom object(s), ideally with `jpg` extension to `./data/raw` directory. (If your objects are simple like ones come with this repo, 20 images can be enough.)
- Resize those photo to uniformed size.
```
python resize_images.py --raw-dir ./data/raw --save-dir ./data/images --ext jpg 
```
Resized images locate in `./data/images/`
- Train/test split those files into two directories, `./data/images/train` and `./data/images/test`

- Annotate resized images with [labelImg](https://tzutalin.github.io/labelImg/), generate `xml` files inside `./data/images/train` and `./data/images/test` folders. 

*Tips: use shortcuts (`w`: draw box, `d`: next file, `a`: previous file, etc.) to accelerate the annotation.*


### Step 2: Open [Colab notebook]
I already uploaded notebook in my file system
https://colab.research.google.com/drive/158IKph6ZvAbNTElOxHTgMrKJ4gXdeQjX#scrollTo=Bq_phVcHnfPl
