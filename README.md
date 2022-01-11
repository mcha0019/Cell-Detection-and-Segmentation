# Cell-Detection-and-Segmentation
This entry to the ISBI 2021 Cell Tracking Challenge was my undergraduate engineering final year project at Monash University (Melbourne, Australia). The core of the developed method is a custom Mask R-CNN neural network that performs cell detections and instance segmentation. The method is smaller, faster and less complex compared to many other methods, and performs well across 14 of the 20 datasets available. 

At the time of submission (Nov 2021), the method ranked 2nd highest in the cell segmentation benchmark in 2 datasets and 3rd in another. Lessons learnt from this initial entry have been applied in an updated version (Jan 2022)

## Background
Since 2012, the Cell Tracking Challenge has been part of the yearly International Symposium of Biomedical Imaging (ISBI), held by the Institute of Electrical and Electronic Engineers (IEEE).  In the challenge, state-of-the-art cell segmentation and tracking methods are tested and compared on 2D and 3D time-lapse microscopy footage from 20 unique datasets with cells of diverse shapes and sizes, captured using various imaging modalities. This data also consists of both real and computer generated simulations.

The challenge itself is split into two benchmarks; the Cell Segmentation Benchmark and the Cell Tracking Benchmark, with each focusing on the corresponding methods. It is popular among researchers, with 60 teams from a broad range of universities and medical research centres worldwide.

### Datasets
The challenge datasets display large variations in the images of cells across their range. Datasets vary significantly, with a wide range of cell types present, each with different shapes and cell features such as the presence of nuclei. Some cells also have complicated features with significant abnormalities. Fluorescence (Fluo), Phase Contrast (PhC), Differential Interference Contrast (DIC) and Brightfield (BF) microscopy modalities were used, resulting in varying image quality (resolution, noise levels, brightness, and contrast levels) across the datasets and modalities. 3D and computer generated datasets are also available, and all images are grayscale.

![Animation](https://user-images.githubusercontent.com/83149912/148882957-1e396299-7775-433d-864b-3a5642d4ae82.gif)

### Final Year Project
In this year's edition of the ISBI challenge, 

>“The primary focus is put on methods that exhibit better generalizability and work across most, if not all, of the already existing datasets, instead of developing methods optimized for one or a few datasets only.”

In keeping with this focus on generalisability, the aim of this project was to develop a method that performs well across the range of datasets and microscope imaging modalities. Additionally, in my submission, an identical method was used on all datasets with no dataset specific tuning, meaning the challenge results give an accurate depiction of the capability of the method across datasets. 

Recent submissions to the challenge have been dominated by the older U-Net architecture (2015) developed specifically for biomedical imaging applications. Instead of developing another U-Net based method, state-the-art Mask R-CNN and DeepLab architectures were used in this project to investigate their potential.

Additional information can be found at the challenge website:
http://celltrackingchallenge.net/

The Cell Segmentation Benchmark:
http://celltrackingchallenge.net/latest-csb-results/

My individual participant page:
http://celltrackingchallenge.net/participants/MON-AU/

## Sample Inferences

## Method
DeepLabV3 and Mask R-CNN networks were trained on 2D and 3D cell segmentation datasets, using silver truth masks for training and gold truth masks for validation. The Mask R-CNN network had better performance, and as it performs object detection and instance segmentation (detects and segments individual cells) compared to DeepLabV3's semantic segmentation (only classifies each pixel in an image as being part of a cell or not), it was the clear choice for the final method.

The final method can be broadly categorised into four main stages. In the first pre-processing stage, images are collated, converted and augmented. 
In the second main stage, images are passed through the Mask R-CNN model, which runs inferences on the images and outputs predicted masks and bounding boxes.
This data is used in the post processing stage, where predictions are refined using techniques such as non maximum suppression to generate the final predictions. 
A final visualisation stage annotates the original images with bounding boxes and/or masks.

![image](https://user-images.githubusercontent.com/83149912/148854190-c488de7b-2199-444b-b322-118694de1ee7.png)

## Results

### Validation Segmentation Metrics

| Metric | | Ver 1, Nov 2021 | Ver 2, Jan 2022 |
| ------------- | ------------- |------------- | ------------- |
| Average Precision	| [ IoU=0.50:0.95 \| area = all ]	| 0.539 | 0.640 |
| Average Precision	| [ IoU=0.50 \| area = all ] | 0.769 | 0.916 |
| Average Precision	| [ IoU=0.75 \| area = all ] | 0.629 | 0.745 |
| Average Precision	| [ IoU=0.50:0.95 \| area = small ]	| 0.430 | 0.615 |
| Average Precision |	[ IoU=0.50:0.95 \| area = medium ]	| 0.687 | 0.699 |
| Average Precision |	[ IoU=0.50:0.95 \| area = large ]	| 0.714 | 0.665 |
| Average Recall | [ IoU=0.50:0.95 \| area = all ]	| 0.587 | 0.680 |
| Average Recall | [ IoU=0.50:0.95 \| area = small ]	| 0.473 | 0.645 |
| Average Recall | [ IoU=0.50:0.95 \| area = medium ]	| 0.765 | 0.739 |
| Average Recall | [ IoU=0.50:0.95 \| area = large ]	| 0.727 | 0.684 |

<sub><sup>These metrics are the same as those used for evaluation in the [Common Objects in Context](https://cocodataset.org/) (COCO) challenge.\
Note: Dataset composition is similar but not identical between versions</sup></sub>

### Official Results

![image](https://user-images.githubusercontent.com/83149912/148864504-929cee59-f109-4647-8c7d-37c72e478cc4.png)
At the time of submission, it was the second best model on the Fluo-C2DL-Huh7 and Fluo-N3DH-CHO datasets, and third on Fluo-C3DH-A549. It was also very close to the top 3 models in BF-C2DL-MuSC, Fluo-N2DH-GOWT1 and PhC-C2DH-U373. 

![image](https://user-images.githubusercontent.com/83149912/148866160-e4c99dbe-a0f5-4f5e-8462-4dfc0d7c2625.png)


