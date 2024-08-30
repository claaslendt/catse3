# Composite activity type and stride-specific energy expenditure estimation (CATSE³)

This repository is dedicated to the research project 'Estimation of activity induced energy expenditure using thigh-worn accelerometry and machine learning approaches'. This project was funded by the Internal Research Funds of the German Sport  University Cologne, grant agreement number L-11-10011-267-121010. Claas  Lendt is further supported by a fellowship of the German Academic Exchange  Service (DAAD).

*All code and data will can be found here upon publication of the corresponding research paper. Stay tuned!*



## Short background

Accurate assessment of energy expenditure (EE) is required to investigate its relationship with health outcomes and to potentially enable individualised energy balance feedback systems. While accurate measurement methods exist in the laboratory and for small-scale research studies (e.g. doubly labelled water or indirect calorimetry), the challenge is to provide accurate estimates of EE in free-living conditions at scale. This is typically done using portable and unobtrusive systems such as accelerometry, but these systems are currently not accurate enough for research and clinical applications. The proposed research project aims to develop and validate a machine learning approach to estimate EE using raw acceleration data collected from the thigh.



## Our study

All sample participants will undergo a standardised activity protocol in the laboratory while EE will be measured using indirect calorimetry. The raw triaxial acceleration of the thigh will be measured and used for subsequent modelling. Activity classification and stride segmentation will be performed on the raw acceleration data using existing algorithmic approaches and subsequently used as model inputs. Model performance will be evaluated against indirect calorimetry and regression models using common aggregated acceleration metrics such as the Euclidean Norm Minus One (ENMO).

In collaboration with the AUT Human Potential Centre, we will implement existing and validated activity classification algorithms. In addition, we plan to merge additional available datasets to enhance our ability to train and test the modelling approach. Based on previous research using activity classification and stride segmentation, we expect the novel modelling approach to perform with greater accuracy compared to commonly used approaches.



![Fig1](https://github.com/claaslendt/thighE3/blob/main/figures/Fig1.jpg)



## Progress of the project :rocket:

(Last update: 2024-08-30)

**We've made a lot of progress. Yay! A quick overview:**

- [x] **Data collection** completed at GSU Cologne.
- [x] **Cleaning and pre-processing** of the collected data.
- [x] **Training of an activity classification model** using a pooled dataset of laboratory and free-living data from three previous studies (n = 69 participants) across a range of different physical activities.
- [x] **Developing the activity-specific stride segmentation approach** for walking, running and cycling.
- [x] **Developing several EE estimation approaches**
  - [x] generic approaches (using ENMO and MAD)
  - [x] activity-specific approach
  - [x] activity- AND stride-specific approach
- [x] **Final analysis and validation** using a hold-out test set
- [x] **Submitting our research** to a scientific journal
- [x] **Publishing the research paper and the data**: the paper has been accepted and is currently awaiting publication!



Some things we are currently planning:

- [ ] Validating our activity classification model on the independent HARTH dataset
- [ ] Validating our CATSE³ algorithm on an independent dataset

 

## Impressions

We are looking forward to share our final approach, the validation results and methods in the near future. Meanwhile, you can find a few impressions below.



**Figure 1**. Confusion matrix for the activity classification model for the validation set.

![confmat_classifier](https://github.com/claaslendt/thighE3/blob/main/figures/confmat_classifier.png)



**Figure 2**. Example of stride segmentation for walking data.

![StrideSegmentationWalking](https://github.com/claaslendt/thighE3/blob/main/figures/StrideSegmentationWalking.png)



**Figure 3**. Preliminary results for the validation data using an activity- and stride-specific LSTM approach.

![Validation_EEStride](https://github.com/claaslendt/thighE3/blob/main/figures/Validation_EEStride.png)
