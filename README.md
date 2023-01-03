EyesMove
==============================

A recurrent neural network model I created & trained, to tell if a person has his eyes closed, based
on EEG data from several electrodes.

This project is a demonstration of creating and training a DNN based on EEG data.

Demo video
------------

### Frontend Overview

![Demo](./demo.gif?raw=true)

Getting Started
------------

From within the repo directory run

`./EyesMove/runner.py`

You can now type in an example number from 3000 patients, and it will return
to the console a percentage of accuracy and the model decision.

-----
About Training & Dataset
--

The dataset was derived from Keras datasets. It consists of thousands of labeled eeg data, which is public domain.
After normalizing it, I used the data without preprocessing it.
link to the dataset:
https://www.kaggle.com/datasets/gauravduttakiit/neuroheadstate-eyestate-classification

Project Organization
------------

    ├── README.md                    <- The top-level README for developers using this project
    ├── LICENSE.md                   <- MIT
    ├── .gitignore                   <- For environment directories
    │
    ├── EyesMove                     <- Containing the software itself
    │   ├── eye_movement             <- Directory of trained model
    │   ├── back.py                  <- backend code
    │   └── runner.py                <- Running the software
    │
    └── tests                        <- Tests directory, .gitignored
        └── backend_tests.py         <- Unit tests of backend
 
Dependencies
------------

- Python
- Pandas
- Keras
- TensorFlow
- NumPy
- Kaggle
--------
# EyesMove
