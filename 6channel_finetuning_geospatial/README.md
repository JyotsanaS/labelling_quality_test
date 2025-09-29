## 6Channel dinov2 model

**Aim:** Generate embeddings for 6 channel geospatial images

**Dataset used for training:**
MS Eurosat data

The model analyzed in this repo classifies images into 10 different classes, the classes are as follows:
* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

On this page, you can learn more about how the model performs on different classes of objects, and what kinds of images you should expect the model to perform well or poorly on.
**Input**: Photo(s) or video(s)
Input shape: 64*64 images

**Output**: The model can detect 10 different image classes

**Channels used(in order):**
Red
Green
Blue
NIR
SWIR1
SWIR2

**Performance:**
* Accuracy: 94.94
* Loss: 0.143

Performance report: https://docs.google.com/document/d/1ydpsOXaunyNCl5yQBeefee9S14dbDJ4mB6SVRXcOR0k/edit?usp=sharing 



