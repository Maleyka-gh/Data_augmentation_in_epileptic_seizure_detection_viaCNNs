
in progress....

## Data Augmentation in Epileptic Seizure Detection based on 3 Acceleration, Heart Rate and Temperature

Epilepsy, characterized by recurrent seizures, poses a significant risk to the individual’s safety. To mitigate
these risks, one approach is using automated seizure detection systems based on Convolutional Neural Networks
(CNNs) which rely on large amounts of data to train effectively. However, real-world seizure data
acquisition is challenging due to the short and infrequent nature of seizures, resulting in a data imbalance.

In this work data augmentation techniques: Standard time series data augmentation techniques and Generative Adversarial Networks
(GANs) - based augmentation are utilized to increase the training dataset for CNNs, aiming for high sensitivity and low false alarm rates in the detection of epileptic seizures. For this purpose, two datasets,

*- one with five features (3D acceleration, heart rate, and temperature)* and another
*- with three features(only 3D acceleration)* are used.

  
 For results comparison, CNN trained without augmented data is used
 as a baseline.

### Data Augmentation
Data augmentation is a technique that artificially increases the size of a dataset by creating
modified versions of existing data. It is particularly helpful in the context of imbalanced
dataset distributions commonly found in real-world applications. It allows additional
data to be generated for underrepresented classes, creating a more balanced dataset.
Studies have shown that data augmentation techniques can improve the generalizability
of deep learning networks, thereby reducing overfitting and enabling the networks to handle
imbalanced datasets more effectively.
These techniques include basic approaches that involve random transformations in
the time, frequency, and magnitude domains and advanced approaches that use Generative
Adversarial Networks (GANs) to generate synthetic time series.

### Short theoretical background of Standard Augmentation techniques

While the choice of data augmentation techniques depends on the
dataset’s properties and the task at hand, several basic techniques have been identified in
the area of time series data. Synthetic samples are generated by applying certain transformations
to the original samples, such as :

*- adding random noise (jittering)*

*- applying warping in time and magnitude (time warping, magnitude warping)*

*- rearranging segments (permutation) within the time series*

*- applying rotation matrices(rotation)* or

*- slicing (window slicing).*

Examples of random-transformation-based (standard) time series
data augmentation techniques are shown in the Figure below. 

<div align="center">
<img src="plots/aug.PNG" alt="overview" width="500"/>
</div>







