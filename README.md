# Kaggle-competition---Predicting-Student-Performance

This is part of the solution of the 8th place in the Predicting Student Performance Kaggle competition. Details of our solution is given here (updating).

To run the training model, firstly, download the datasets
```
!kaggle competitions download -c predict-student-performance-from-game-play
!kaggle datasets download -d shinomoriaoshi/pspext-data
!kaggle datasets download -d shinomoriaoshi/pspsuplementary-data
!unzip predict-student-performance-from-game-play.zip -d data
!unzip pspext-data.zip -d ext_data
!unzip pspsuplementary-data -d ext_data
```

After that, clone this repository, and move the data into the code directory
`
mv ext_data Kaggle-competition---Predicting-Student-Performance
mv data Kaggle-competition---Predicting-Student-Performance
`

Install polars if you don't have it `!pip install polars`

Then, train the models,
`
python Transformer/train.py
`
