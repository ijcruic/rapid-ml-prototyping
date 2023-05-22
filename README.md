# Code for _Rapid Prototyping of ML Solutions_

![Rapid ML Prototyping](project_picture.png)

[![Last Commit](https://img.shields.io/github/last-commit/ijcruic/rapid-ml-prototyping)][https://github.com/ijcruic/rapid-ml-prototyping]
[![Repo Size](https://img.shields.io/github/repo-size/ijcruic/rapid-ml-prototyping)][https://github.com/ijcruic/rapid-ml-prototyping]

The overall objective for this codebase is to give you instruction on machine learning such that you, in the course of the work in your field, can quickly apply machine learning tools to your problems. It is not meant to make one an expert in data science or to teach the foundations of data literacy, but rather teach a domain expert (i.e. someone who does not work in data science for a living) how to quickly implement machine learning solutions for possible use cases in their domain work roles. The presented techniques can also be useful for seasoned data scientists as well :smiley:. 

## Contents of this Repository
The contents of this repository are the code and slides for instruction on how to implement fats ML solutions. For the code, the files are broken down by the main type of data domain (e.g., image, text, and tabular). Files are named by when they were presented, what they were presented for, and a name. For example, a jupyter notebook that details how to implement a tabular solution taught at conference x in January of 2023 would be `01JAN23_Conference X_Tabular.ipynb`. The following is a current list of contents of the repository

- Slides
    1. [29NOV22_AvengerCon VII_Rapid ML Prototyping](Slides/29NOV22_AvengerCon VII_Rapid ML Prototyping.pdf): Slides from [AvengerCon VII](https://avengercon.com/) conference workshop. The workshop covered creating fast supervised ML solutions for Text, Tabular, and Image
    
- Tabular
    1. [29NOV22_AvengerCon VII_Tabular](Tabular/29NOV22_AvengerCon VII_Tabular_Complete.ipynb): Complete and work through versions of implementing a fast tabular solution. The data set is a water quality data set from a Kaggle competition. The notebook features EDA, data processing (including imputation for missing data), feature engineering, and hyperparameter tuning. The main ML model used is [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.5/).
    2. [2MAR23_Playground Series 3_8](Tabular/2MAR23_Playground Series 3_8.ipynb): A complete example of implementing a fast tabular solution. The data set is a Kaggle playground series on gem quality rating ([Season 3, Episode 8](https://www.kaggle.com/competitions/playground-series-s3e8)). The notebook features EDA, data processing (including imputation for missing data), feature engineering, and hyperparameter tuning. The main ML model used is [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.5/).

- Text
    1. [29NOV22_AvengerCon VII_Text_Complete](Text/29NOV22_AvengerCon VII_Text_Complete.ipynb): Complete and work through versions of implementing a fast text solution. The data set is a Kaggle competition dataset around classifying text quality. The notebook features some text EDA, a brief demonstration of zero-shot text labeling, and finetuning a model from [HuggingFace](https://huggingface.co/docs/transformers/training).

- Image
    1. [29NOV22_AvengerCon VII_Vision_Complete](Image/29NOV22_AvengerCon VII_Vision_Complete.ipynb): Complete and work through versions of implementing a fast Image solution. The data set is a custom one, compiled from various sources, centered around finding images of military vehicles from a bunch of online images. The notebook features some brief image EDA and finetuning a [timm](https://timm.fast.ai/) model using [FastAI](https://docs.fast.ai/tutorial.vision.html).

#### Some Tips...

:white_medium_square:__Narrowly and specifically design the problem__

:white_medium_square:__Use existing code/models/workflows/insights whenever possible. In particular, test to see if there is a general purpose model or zero-shot model which can deliver the performance you need, without needing to do any training__

:white_medium_square:__Start simple and basic, and build more complexity over iterations__

:white_medium_square:__Stay [data-centric](https://github.com/daochenzha/data-centric-AI) in your approach__