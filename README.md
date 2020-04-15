# EEG
Motor imagery decoding from EEG data


# Implementation of tools

## `pip`
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --user
```

## Installation of libraries
```
pip install mne matplotlib sklearn
```

MNE (https://mne.tools) is a library dedicated to the visualization and analysis 
of MEG, EEG, sEEG, ECoG signals.

matplotlib (https://matplotlib.org) is a scientific visualization library for 
producing quality images (especially for scientific publications).

sklearn (https://scikit-learn.org) is a library for machine learning.

---

# The dataset

The data set used contains EEG data acquired during different mental motor imagery (IMM) tasks.
The 109 subjects performed 4 types of IMM tasks for 4 seconds, imagine:

- move the left hand
- move the right hand
- move both hands simultaneously
- move both feet simultaneously

64 EEG channels were recorded at 160Hz.

---


## The model

We will try to discriminate between the classes moving the right hand, 
and moving the feet. To do this, we use the algorithms of CSP (Common Spatial Patterns) 
and LDA (Linear Discriminant Analysis).

The different stages of treatment are:

- Filter the signal between 8 and 30 Hz
- Extract the windows linked to the events: move the right hand and the feet.
- Create the CSP + LDA model

## Model evaluation

- Evaluate the model by cross validation

We will identify a subject for whom cross-validation tends to show that it works 
and another for whom it does not work.

For each of these two subjects, display the patterns selected by the CSP.


## Removal of artifacts

Use the ICA to eliminate windows contaminated with blinking.

