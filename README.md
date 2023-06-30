# Kaggle-competition---Predicting-Student-Performance---8-th place solution

This is part of the solution for the 8th place in the Predicting Student Performance Kaggle competition. Details of our solution are given here (URL updating).

To run the training model, firstly, download the datasets
````
kaggle competitions download -c predict-student-performance-from-game-play
kaggle datasets download -d shinomoriaoshi/pspext-data
kaggle datasets download -d shinomoriaoshi/pspsuplementary-data
unzip predict-student-performance-from-game-play.zip -d data
unzip pspext-data.zip -d ext_data
unzip pspsuplementary-data -d ext_data
````

After that, clone this repository, and move the data into the code directory
`
mv ext_data Kaggle-competition---Predicting-Student-Performance----part-of--8th-solution
mv data Kaggle-competition---Predicting-Student-Performance----part-of--8th-solution
cd Kaggle-competition---Predicting-Student-Performance----part-of--8th-solution
`

Install polars if you don't have it `!pip install polars`

Then, train the models,

`
python Transformer/train.py
`

After training the model, please infer the embedding by changing the attribute `mode` value to `infer_embedding` and run `!python Transformer/train.py` again. After that, run `!python XGBoost/train.py`. If you get errors, please move the folder `feature_lists` to `model/v7/b` folder, and run `!python XGBoost/train.py` again.

The code however could be problematic due to the cleaning process. In case you have any questions or feedback, please contact me via phanminhtri2611@gmail.com
