Epilepsy is a widespread disease characterized by repeated epileptic seizures, which can
pose a serious risk to the individual’s safety and lead to life-threatening injuries. One approach
to reduce this risk is by using automated seizure detection systems, which can alert
caregivers or medical professionals to the occurrence of a seizure. State-of-the-art seizure
detection systems, such as those based on Convolutional Neural Networks (CNN), rely on
large amounts of data to train effectively. However, obtaining real-world seizure data is
a costly and time-consuming process, and the duration of actual seizure activity during
extended EEG recordings is typically quite short, often lasting only seconds or minutes.
As a result, the available dataset for training these systems is often highly imbalanced,
with far more non-seizure data than seizure data.

In this work, data augmentation techniques are utilized to increase the data available for
the MOND project and train the CNN for seizure
detection, with more available data to achieve high sensitivity and a low false alarm rate in
the detection of epileptic seizures with motor components. Therefore, two datasets are created
from the original data: one with five features (3D Acceleration in x,y,z direction, heart rate and
temperature) and the other with only three features (only 3D Acceleration).
Standard time series data augmentation techniques and Generative Adversarial Networks
(GANs) -based augmentation is applied to the seizures available inside dataset.
The CNN trained without augmented data is used as a baseline to compare improvements
that have been made using the augmented data.
