# Deep Learning Classifier Tuning

## Overview

The whole study focuses on extensively experimenting and finetuning Neural Netwwork classifier by looping over different settings for parameters involved.

The primary evaluation criteria for this study is taken as Cost Matrix. The lower the cost of missclassification, better the model.

The other metrics considered are: Loss, Accuracy, recall, precision, AUC.

## Features

For the whole study, includes detailed pre-processing of data.

Following model and finetuning is followed:
- Model 1: Initial Model: 1 layer with 120 neurons, optimizer 'sgd', iterations=200, lr = 0.0001.
- Model 2: FineTuning No. of neurons: Test 4 different settings for Neurons as 30, 60, 180, and 240.
- Model 3: FineTuning No. of hidden layers: Test 4 different settings for layers as (30, 30), (60, 30), (60, 60) and (120, 60, 30).
- Model 4: FineTuning No. of max iterations: Test 4 different settings for max iterations as 200, 400, 600, 800.
- Model 5: FineTuning learning rate: Test 4 different settings for learning rate as 0.002, 0.004, 0.006, 0.008.
- Model 6: FineTuning Optimizer: Test the other optimizer setting of 'adam'.

For final Comparative Analysis, please refer to last section in the Jupyter Notebook.

## Usage

The best way to see the models in action is to clone the repository.

Be mindful, you might need install modules as per your environment. Best way to do this would be to run this in your terminal:
```Terminal
pip install <module>
```

## Contact
Thank you for dropping by, for any queries please feel free to contact on LinkedIn or by email

[LinkedIn](https://www.linkedin.com/in/nikhil-arora-6837501a4/) | [Email](nikhil.wm27@gmail.com)

Please have a look at my other Projects:

[GitHub](https://github.com/nikhil-hs27/Projects)