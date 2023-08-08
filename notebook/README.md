## Notebooks

Please put your Jupyter notebooks in this folder.
A good idea is to name them following a numbering order. For instance:

    1.1_explore_data.ipynb
    1.2_data_cleaning.ipynb
    1.2.1_data_cleaning_evaluation.ipynb
    2.1_machine_learning_model.ipynb

You can use the following Python code to be able to import the Python files you have put in `src`:

    import sys
    sys.path += ['../src/']


All code dependencies (Python packages) should be specified by the conda environment described in `../environment.yml`.
You can create this conda environment with this command:

    conda env create --file ../environment.yml

After creating it, you can activate and add this environment to the global Jupyter with

    conda activate PROJECT-NAME
    ipython kernel install --name "PROJECT-NAME"

If you modify the `environment.yml` file, then you can update the (existing) conda enviroment by using:

    conda env update --file ../environment.yml --prune
