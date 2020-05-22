## How to Run

### Install required libraries
`pip3 install -r requirements.txt`



### Data Preperation
- I preapred data from script where i used method to fimd  xmin, xmax, ymin, and ymax using this formula

```
x = xmin
y = ymin
w = xmax - xmin
h = ymax - ymin

```

### detection network

- I used SSD with tensorflow API

### hyper-parameters for anchor box tuning.

```
	num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
```
        
###  Traing Notebook
- training.ipynb in this notebook i traoined model for object dtection where i used tensorflow api with SSD.


### Cheaking Result

- run inference.py for count objects in images.
- After running successful one json file will generate, Where you can see your result.

```

     python inference.py

```







