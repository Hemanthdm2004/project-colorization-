<div align="center"><img src="logo.png" /></div>
<hr />
<div align="center">
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /></a>
<a href="https://heroku.com/"><img src="https://img.shields.io/badge/heroku%20-%23430098.svg?&style=for-the-badge&logo=heroku&logoColor=white" href="https://heroku.com/" /></a>
</div>



<br />

<div align="center"><h4> An image colourisation model trained on 570k images using autoencoders to colourise grayscale images.</h4></div>
<div align="center"><a href="https://image-colouriser-streamlit.herokuapp.com/"><img src="https://forthebadge.com/images/badges/check-it-out.svg" /></a></div>

<br />

<div align="center"><img src="demo.gif"></div>

## Technology Stack
- [Pytorch](https://pytorch.org/) for building the model
- [Streamlit](https://www.streamlit.io/) for building the web application
- [Heroku](https://heroku.com/) for deploying the web application

## To Run the Notebook using Pretrained Weights

The path files for the models trained on landscapes, people, fruits, and animals are available as [landscapes.pth](https://github.com/Priyansi/image-colouriser-streamlit/blob/master/app/landscapes.pth), [people.pth](https://github.com/Priyansi/image-colouriser-streamlit/blob/master/app/people.pth), [fruits.pth](https://github.com/Priyansi/image-colouriser-streamlit/blob/master/app/fruits.pth) and [animals.pth](https://github.com/Priyansi/image-colouriser-streamlit/blob/master/app/animals.pth).




### Datasets 
The following datasets were used to train the respective models-
#### Landscapes
1. [ImageNet](https://www.kaggle.com/lijiyu/imagenet)
2. [Flickr](https://www.kaggle.com/hsankesara/flickr-image-dataset)
3. [Landscape Classification](https://www.kaggle.com/huseynguliyev/landscape-classification)
4. [Scene Classification](https://www.kaggle.com/nitishabharathi/scene-classification)
#### People
1. [Clothes Classification](https://www.kaggle.com/salil007/caavo)
2. [CelebA Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)
#### Fruits -
1. [Fruits360](https://www.kaggle.com/moltean/fruits)
2. [Fruit Recognition](https://www.kaggle.com/chrisfilo/fruit-recognition)
#### Animals -
1. [Animals10](https://www.kaggle.com/alessiocorrado99/animals10)
2. [Arthropod Taxonomy Orders Object Detection Dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset)
3. [Animal Faces](https://www.kaggle.com/andrewmvd/animal-faces)
4. [African Wildlife](https://www.kaggle.com/biancaferreira/african-wildlife)
5. [Animals Dataset](https://www.kaggle.com/navneetsurana/animaldataset)
6. [The Oxford-IIIT Pet Dataset](https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset)
