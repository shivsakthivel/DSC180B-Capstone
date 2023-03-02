### Exploring the viability of Convolutional Neural Networks (CNNs) on a multi-label classification task to simultaneously detect Pulmonary Edema and Pleural Effusion

### INTRODUCTION

In this project, the MIMIC-CXR database of chest radiographs and patient reports will be used to train a CNN classifier, using transfer learning techniques. The MIMIC-CXR database consists of 377,000 radiographs, stored in a DICOM format with resolutions of up to 4K. As a result, the size of this database is 4 TB, and it is currently hosted in a Google Cloud bucket, where credentialed users can access the data. For this project, the classifier will predict two labels, associated with Pulmonary Edema and Pleural Effusion. In its current state, the database metadata contains information associated with 14 labels, automatically produced by running both the CheXpert and NegBio algorithms on the patient reports. Considering the performance of the model and computational resources available, the inputs to the classifier will be downsampled to standard 512*512 images, and a subset of 34,000 images which have labels associated with both conditions will be used for the training process. 

The goal is to build the classifier and analyze its performance on unseen data. The evaluation will be performed by assessing the accuracy metric on each label, analyzing the training and validation loss curves to test if the model is behaving correctly, analyzing the AUC-ROC curve for each label as a measure of the model’s diagnostic ability, and finally the activation maps on the radiographs as a test of the model’s interpretability.

### METHODS

#### Transfer Learning
The major technique used in the model training process is Transfer Learning. Transfer learning is a machine learning technique where a pre-trained model on a large dataset is fine-tuned on a smaller related dataset. The goal of transfer learning is to leverage the knowledge learned from the large dataset and apply it to the smaller related dataset, allowing the model to learn more efficiently and perform better. The key advantage of transfer learning is that it saves the time and computational resources needed to train a deep learning model from scratch, and it also helps the model to overcome the problem of limited data by leveraging knowledge from a similar domain.

#### Oversampling to resolve Class Imbalance
For individual labels, the dataset has roughly the same number of positive and negative cases. However, within the selected subset, there is a bias towards the number of cases with the same values on both labels (positive or negative on both Edema and Effusion), and against those with opposite labels on both conditions. As a result, the model more often than not, can be expected to produce correlated outputs, which might defeat the purpose of performing multi-label classification, where the outputs are expected to be independent of each other, at least in the prediction process. Therefore, there can be two approaches to handling this imbalance. One method would be to adjust the weights associated with the binary cross entropy loss function used in the model evaluation, and assign higher weights to cases with opposite labels to balance the number of cases on the other side. However, with both labels being produced independently, the process of model compiling does not account for four separate classes ((0, 0), (0, 1), (1, 0) and (1, 1) as opposed to just two classes on two variables. Therefore, in this project, the model will be trained on a random oversampling of the rarer classes of images on a scale proportional to its relative rarity in the dataset.

### MODEL TRAINING PIPELINE
Figure 1
! [Pipeline](/DSC180B-Capstone/docs/assets/pipeline.png)


### RESULTS

Table 1: Results of the Multi-Label Classifier on the Test Dataset:
| | Pulmonary Edema | Pleural Effusion |
| --- | --- | --- |
| Test Accuracy | 78.9% | 81.0 % |
| ROC AUC | 0.868 | 0.885 |

Table 2: Results of the separately trained binary classifiers on the Test Dataset:
| | Pulmonary Edema | Pleural Effusion |
| --- | --- | --- |
| Test Accuracy | 79.4% | 81.4 % |
| ROC AUC | 0.875 | 0.901 |

### CONCLUSION

