# essab.github.io
Visual defect detection for manufacturing

# Visual Defect Detection for Manufacturing — Project Proposal

## Introduction and Background

It is essential for manufacturers to catch defective products automatically, but since the defects are rare, most inspection data only contains good parts.

### Literature Review

In manufacturing factories, it’s difficult to teach AI to identify flawed products, because it rarely sees defects. Stojanovic et al. clustered 3D-scan deviations of Whirlpool fan blades to learn which deviations are normal and they flagged the ones that were not[1]. Lee et al. trained an LSTM autoencoder on normal sensor data and flagged high reconstruction error, finding that supervised models need costly labels while unsupervised ones are hard to validate [2]. Ko et al. labeled engines using only after-sales repair records and used one-class classifiers because, as we mentioned, the defects are rare[3]. In combination, previous research uses sensor and 3D data, so we will test the same trade-off on images and measure exactly how many labels are needed for supervised detection catch up with unsupervised.

### Dataset Description

MVTec AD contains over 5000 high-resolution images split between 15 categories: 5 textures and 10 objects. Each category has three folders: defect-free images, test images split by defect category, and pixel-level masks of each defect. Grid, screw, and zipper images are grayscale.

**Dataset Link:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## Problem Definition

### Problem

Manually inspecting products for scratches, cracks, dents, or missing parts is slow, expensive, and inconsistent when many products need to be checked. Also, defective products are generally less common, which makes it difficult to collect enough labeled defect images for training a machine learning model. This project aims to develop an automated visual defect detection system that can identify defective products from images, even when only a small number of defective examples are present.

### Motivation

The main motivation is to reduce the dependence on manually labeled defect data. We will first build unsupervised methods that learn how a normal product looks and flag unusual images as defects, then compare them with supervised models trained on labeled good and defective images. This comparison will show when unsupervised learning is enough and how much labeled data supervised models need to perform better.

## Methods

### Data Preprocessing Methods

- **Resizing, channels, normalization:** resize, convert grayscale images to three channels, and apply ImageNet normalization.
- **Feature extraction:** a frozen pretrained ResNet-18 turns each image into a 512-dimensional pooled vector.
- **Scaling and dimensionality reduction:** StandardScaler, then PCA keeping about 95% of variance
- **Augmentation and splitting:** defect images are split into train and test folds, and only the training folds are augmented, so test images never leak into training.

### Unsupervised Learning

- **PCA reconstruction error** (`sklearn.decomposition.PCA`): Following [2], PCA learns how normal parts vary, so defects reconstruct poorly and get a high anomaly score.
- **Gaussian mixture model** (`sklearn.mixture.GaussianMixture`): we do soft clustering with probabilities and parts with low probability under the normal clusters are flagged as defects.
- **K-Means distance** (`sklearn.cluster.KMeans`): According to [1], we cluster normal parts, and the distance to the nearest center is the anomaly score.
- **Patch nearest neighbors** (`sklearn.neighbors.NearestNeighbors`): get scores and produce heatmaps based on nearest neighbors.

### Supervised Learning

- **SVM** (`sklearn.svm.SVC`): maximum-margin boundaries because it works well for small subset of examples.
- **Random forest** (`sklearn.ensemble.RandomForestClassifier`): for observing nonlinear patterns and seeing which feature matters the most. 
- **Fine-tuned ResNet-18** (PyTorch): adapts the pretrained network, which may be most advantageous with labeled data.

**Main experiment.** Plot supervised AUROC after training on 1, 2, 5, 10 and all defects per category vs the unsupervised scores.

## Potential Results and Discussion

### Quantitative Metrics

We will evaluate the models using Image-level AUROC, accuracy, precision, recall, F1-score, and Average Precision (AP). AUROC will be the primary metric for anomaly detection, while precision and recall will measure how well defects are identified. F1-score and AP will handle the class imbalance, and Macro-F1 will be used to evaluate performance across different defects/categories equally.

### Project Goals

Main goal of this project is to develop a system that can detect defects correctly while requiring as little labeled defect data as possible. We will compare different unsupervised methods with supervised machine learning and deep learning models. The project aims to reduce manual inspection effort and make quality control faster and more consistent. Since failing to identify a defective product could have serious consequences, we will also take into account ethical concerns like false negatives.

### Expected Results

We expect supervised models to perform better when sufficient labeled defect images are available, while unsupervised methods should be useful when labeled defects are very limited. We also expect deep learning models to provide stronger performance than simpler machine learning methods, particularly for complex visual defects. We also expect performance to vary depending on the type and visibility of the defect.

## References

[1] L. Stojanovic, M. Dinic, N. Stojanovic, and A. Stojadinovic, "Big-data-driven anomaly detection in industry (4.0): An approach and a case study," in *Proc. 2016 IEEE Int. Conf. Big Data (Big Data)*, Washington, DC, USA, 2016, pp. 1647–1652, doi: 10.1109/BigData.2016.7840777.

[2] K. S. Lee, S. B. Kim, and H.-W. Kim, "Enhanced anomaly detection in manufacturing processes through hybrid deep learning techniques," *IEEE Access*, vol. 11, pp. 93368–93380, 2023, doi: 10.1109/ACCESS.2023.3308698.

[3] T. Ko, J. H. Lee, H. Cho, S. Cho, W. Lee, and M. Lee, "Machine learning-based anomaly detection via integration of manufacturing, inspection and after-sales service data," *Industrial Management & Data Systems*, vol. 117, no. 5, pp. 927–945, 2017, doi: 10.1108/IMDS-06-2016-0195.
