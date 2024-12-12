# README

---

At the start, most of my data preprocessing and model training were done in python scripts. Once all the data was fetched and the baseline was trained, I decided to do more formal preprocessing in the `DataPreprocessing.ipynb` notebook and training in the `Model.ipynb` notebook.

To run any of the scripts, it should be as easy as running a command like:

`python3 baseline_model_training.py`,

although you might run into issues with Python packages. 

I use an Anaconda distribution of Python. The only packages I had to add were `torch` and `pandas_ta`

For more information on installing `pandas_ta`, see the documentation here: [github.com/pandas-ta](https://github.com/twopirllc/pandas-ta)

Python scripts can be found in the `scripts` directory, and Jupyter notebooks can be found in the `notebooks` directoy. Data is in the `data` directory, figures are found in the `figures` directory, and final models are in the `models` directory.