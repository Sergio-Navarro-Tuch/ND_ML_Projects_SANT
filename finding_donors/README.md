This folder contains the report on the CharityML project for Machine Learning Nanodegree.
The contained files are as follows:
>- finding_donors.ipynb.- Project Notebook with all code cells excecuted
>- report.html.- finding_donors notebook's HTML export

The notebook was run on a python 3.6 environment

## Versioning

v1.1

## Updates

The folder now contains the updated version of the project Charity ML with the following corrections:
>- **Evaluating Model performance**.- Fbeta Score formula correction to consider a beta of 0.5 instead of 1
>- **Evaluating Model performance**.- Setting of the `random_state`parameter of the models to `42`in order to allow the replication. Also provide a justification for the presetting of the `gamma`parameter of the SVC to `'scale'`.
>- **Improving Results**.- Attempt on simplifying the explanation of the algorithm for **Question4**
>- **Improving Results**.- Setting of the `random_state` parameter of the final model to `42`in order to allow replicability.
>- **Improving Results**.- **Question 5** table updated after Naive predictor Fbeta score formula correction and comparison with updated results.


## Known remaining issues

>- Even after the Fbeta score formula correction, the resulting value obtained is of `0.2756` instead of the expected `0.2917`of the reviewer. I am still unsure for the cause of such discrepancy.

## Authors

* **Sergio Navarro** - *Initial work* - [Sergio-Navarro-Tuch](https://github.com/Sergio-Navarro-Tuch)

## License

N/A

## Acknowledgments

* [Udacity Machine Learning](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)
