# EvaluationOf_BreastCancerData_KNN

EVALUATION OF BREAST CANCER DATA WITH MACHINE LEARNING TECHNIQUES

(This study is a graduation project of Sakarya University Computer Engineering Department.)

This study was prepared with the aim of diagnosing cancer cells with Machine Learning by considering the Breast Cancer dataset. In the study, it is aimed to correctly evaluate the diagnostic feature of the computer in the data set by using the Python programming language.

<img width="878" alt="d1" src="https://user-images.githubusercontent.com/63105388/215253682-44073896-c1b2-4546-98d6-36fab5215402.png">

In the first part of the study, the data set was examined and the problem was defined. Then, a detailed analysis was carried out with Exploratory Data Analysis (EDA). Outliers were identified and these values were removed from the data set. Finally, the dataset was divided into two parts as training and test dataset, and after this point, the dataset became trainable.

<img width="1044" alt="PairPlot" src="https://user-images.githubusercontent.com/63105388/215253735-e839d2c0-6a41-47e7-9b93-3076cc94852f.png">

<img width="750" alt="Screen Shot 2021-04-25 at 6 14 21 PM" src="https://user-images.githubusercontent.com/63105388/215253751-2acf2266-3812-4ce7-ba80-62c815625ca9.png">

After these processes, KNN algorithm training was carried out and 95% accuracy was obtained as a result of this process. This value is higher than expected because no overfit or underfit has occurred. Then, the process of finding the best parameters was done, where over-learning took place and therefore the accuracy rate decreased to 94%.

With PCA analysis, the data set was reduced to two dimensions, but the accuracy value decreased to 92%.

<img width="615" alt="Screen Shot 2021-05-30 at 10 48 14 PM" src="https://user-images.githubusercontent.com/63105388/215253810-3801612e-eea7-46fd-a444-a84ecfc21890.png">

After the NCA analysis, the data set was again reduced to two dimensions, but a good model was obtained without over-learning or under-learning. As a result of this process, the accuracy value was determined as 99%, and the machine learned the data set as desired.

<img width="452" alt="Screen Shot 2021-05-31 at 2 27 57 PM" src="https://user-images.githubusercontent.com/63105388/215253943-9985c420-d148-430a-ad90-967325f655f5.png">
