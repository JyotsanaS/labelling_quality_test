# Labelling-Quality-Benchmark-Suite
This repository contains all the experimental methods for evaluating Labelling Quality Test. 

Documentation:-
---------------
1. Literature Survey, Plan
https://docs.google.com/document/d/1iUR7Yewi3Yb6u8Wz0kcTswRR1C46gHH_JqNqUZCaNVA/edit?usp=sharing

Embedding Generation:-
----------------------

- ROI generation<br>
     1. Resizing<br>
     2. Resizing preserving aspect ratio, padding the remaining pixels with 0, centering the image<br>
     3. Padding the original image, keeping the image centered. In case the RoI is larger, it the step would be like 2<br>
     4. Center cropping the minimal area around the original image, to extract a square image and then resizing it<br>
- Embedding Generation<br>
     1. DinoV2 (no training)<br>

LQC Algorithms Supported:-
------------------------
**1. Embedding Based**<br>
     - Single Centroid based Method<br>
     - FINCH based clustering<br>
     - KNN based entropy<br>
     - Autoencoder per class with rereconstruction loss<br>
**2. Training Based**
     -

Datasets:-
----------
[To Do: Push to s3,replace this]
raga-basic
pascal:
/home/ubuntu/LQC/dataset/pascal/VOC2012/pascal_trainval
mscoco:
/home/ubuntu/LQC/dataset/mscoco/val2017
