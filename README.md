# Twin (Siemese) Nuetral Network
 
## Overview
This project implements a Twin Neural Network (also known as a Siamese Neural Network) designed to determine the similarity between two input samples. The architecture is especially useful for tasks where comparing inputs is more important than classifying them individually.
 
## Data 
We utilized 119 user signature samples from the ICDAR 2011 Signature Verification Datasets, sourced via Kaggle, which include 2,207 genuine and 2,082 forged signatures. All images were resized to 105×105 pixels, though this preprocessing step may introduce slight distortions to the signature patterns.
### Note
Due to our further exploration on our own signitures and professors we are not including the data folders. Here is the link to the original kaggle dataset- https://www.kaggle.com/datasets/mallapraveen/signature-matching
 
## Architecture
### Dual Inputs: Two inputs are passed to the model simultaneously.
 
### Parallel Subnetworks: Each input is processed by an identical subnetwork (shared weights), running in parallel.
 
### Binary Classification: The resulting distance is passed through a sigmoid layer, and we use Binary Cross Entropy as the loss function.
 
<img src="https://github.com/user-attachments/assets/476733b3-7c0c-4ceb-901e-cd4d23427700" width=500 />
 
## Initial Results
Our initial results showed an average accuracy of 99.40% on the provided test set. However, the model tended to classify genuine signatures as forged more often than the reverse. However, it's preferable to falsely reject your own signature than to mistakenly accept a forged one.
 
### Note
All the training and testing images were of similar quality and style, and our small dataset size likely contributed to the high accuracy, possibly due to overfitting. With medium to large datasets, we would expect the accuracy to drop to 80–90% or even 70–80%.

<img src="https://github.com/user-attachments/assets/b0572835-f752-4e3a-9d57-8421fb45b172" width=500 />
 
## Further Exploration
During testing, we created 16–20 genuine and 6–8 forged signatures by writing with both pen and pencil. Introducing this more varied and realistic data significantly reduced the model's accuracy, highlighting its sensitivity to changes in input quality. Despite the drop, the model still detected about a quarter of forged attempts, even though the forgeries involved closely tracing each other’s signatures. Simple signature styles led to false rejections, flagging all of one participant's genuine signatures as forgeries. This points to limitations in the model’s ability to generalize, likely due to a lack of data augmentation.
 
### Note
For security and privacy reasons, we are not including samples of our own signatures in this report.
