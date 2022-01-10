# Cell-Detection-and-Segmentation
This entry to the ISBI 2021 Cell Tracking Challenge was my undergraduate final year project at Monash University (Melbourne, Australia). The core of the developed method is a custom Mask R-CNN neural network that performs cell detections and instance segmentation. The method is smaller, faster and less complex compared to many other methods, and was trained and performs well across 14 of the 20 datasets available. 

At the time of submission (Nov 2021), the method ranked 2nd highest in the cell segmentation benchmark in 2 datasets and 3rd in another. Lessons learnt from this initial entry have been applied in an updated version.

## Background
Since 2012, the Cell Tracking Challenge has been part of the yearly International Symposium of Biomedical Imaging (ISBI), held by the Institute of Electrical and Electronic Engineers (IEEE).  In the challenge, state-of-the-art cell segmentation and tracking methods are tested and compared on 2D and 3D time-lapse microscopy footage from 20 unique datasets with cells of diverse shapes and sizes, captured using various imaging modalities. This data also consists of both real and computer generated simulations.


The challenge itself is split into two benchmarks; the Cell Segmentation Benchmark and the Cell Tracking Benchmark, with each focusing on the corresponding methods. It is popular among researchers, with 60 teams from a broad range of universities and medical research centres worldwide.

Final Year Project
In keeping with the primary focus on generalisability of the 2021 Edition of the challenge, the aim of this project was to develop a method that performs well across the range of datasets. Recent submissions to the challenge have been dominated by the older U-Net architecture, so the state-the-art Mask R-CNN and DeepLab architectures were used instead.

Additional information can be found at the challenge website:
http://celltrackingchallenge.net/

The Cell Segmentation Benchmark:
http://celltrackingchallenge.net/latest-csb-results/

My individual participant page:
http://celltrackingchallenge.net/participants/MON-AU/

Official benchmark results prove both the capability and potential of the submitted method based on these newer models, achieving competitive scores in a range of datasets, including two 2nd places and one 3rd place in the global dataset rankings. 
