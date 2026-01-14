# Deep-Learning-Final-Project-in-Multi-retinal-Disease-Detection
In medical imaging, obtaining large-scale, labeled datasets is often challenging due to privacy concerns, high annotation costs, and limited availability of expert knowledge. To effectively learn and boost performance on small-scale datasets, we leverage transfer learning techniques which consist of models that are trained on large amounts of data.

## Goal

To improve the performance of multi-label retinal image classification using transfer learning by fine-tuning models, while deepening the understanding of deep learning techniques.

## What was Done?

In this project, we address the problem of multi-label retinal disease detection, focusing on three major conditions: Diabetic Retinopathy (DR), Glaucoma (G), and Age-related Macular Degeneration (AMD). To tackle the challenge of limited annotated medical data, we adopt transfer learning strategies, leveraging models pretrained on large-scale datasets and fine-tuning them for multi-label retinal image classification. The experiments are conducted on the ODIR dataset, which is divided into a training set of 800 images, a validation set of 200 images, an offsite test set of 300 images, and an onsite test set of 250 images, with all images standardized to a resolution of 256×256. The evaluation metrics include precision, recall, F-score of each disease and the average F-score over the three diseases.
