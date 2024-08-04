# HMS Harmful Brain Activity Classification
## Overview
This project aims to classify harmful brain activity patterns using spectrograms derived from EEG waveforms. The dataset includes both Kaggle-provided spectrograms and raw EEG waveforms, with a focus on combining both types of data for improved model performance.
## Dataset
- Kaggle Spectrograms: 10-minute long spectrograms provided by Kaggle.
- EEG Waveforms: 50-second raw EEG waveforms, with the middle 10 seconds overlapping with the Kaggle spectrograms.
- Spectrograms Dataset: Includes spectrograms created from raw EEG data using a novel formula. Available in the Kaggle dataset.
- Train Data: Includes labels for different brain activity types such as seizure, LPD, GPD, LRDA, GRDA, and other.
- Evaluation: Match the given probability distribution, provided by experts in the field, with your own distribution. The closeness of the match will be measured using the KL divergence metric.
<img src="https://storage.googleapis.com/kaggle-media/competitions/Harvard%20Medical%20School/eFig2.png"></img>
## Project setup
- Hardware: Utilizes 2x T4 GPUs with mixed precision enabled for efficient training.
- Libraries: TensorFlow 2.13.0, EfficientNet, Albumentations for data augmentation.
- Competition link: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
## Data Processing
- Spectrograms Loading: Efficient loading and processing of spectrograms from large datasets.
- EEG Spectrograms: Generated from raw EEG data and included in the Kaggle dataset.
- Data Augmentation: Includes horizontal flipping and coarse dropout to enhance model robustness.
## Model Architecture
- Base Model: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4 used for ensembling.
- Input: 128x256x8 images combining Kaggle and EEG spectrograms.
- Training Strategy: Includes a step learning rate schedule with different learning rates for different epochs.
- Used kl-divergence as a loss function
## Performance
- Cross-Validation Score: Achieved 0.431976 with ensemble using both spectrograms.
- Leaderboard Score: Achieved 0.500403 in the competition leaderboard.
## Training
- Data Loader: Handles non-overlapping EEG IDs, extracts and standardizes spectrograms, and prepares batches for training.
- Training Schedule: Uses a step learning rate schedule with a base learning rate of 1e-3 and adjustments for subsequent epochs.
## Dependencies
Install required packages using ```pip install tensorflow efficientnet albumentations```
## Acknowledgments
- Thanks to Kaggle for providing the datasets.
- Special thanks to contributors for their datasets and tools.
- Special thanks to [Chris Deotte](https://www.kaggle.com/cdeotte) for sharing his starter notebooks.

