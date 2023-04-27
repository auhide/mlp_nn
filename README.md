# Multilayer Perceptron API Implemented From Scratch With NumPy
The project was previously deployed on Heroku but currently is not deployed anywhere. The repository of the UI is [this one](https://github.com/auhide/nnvis-ui).

You can see a walkthrough video of the full project [here](https://youtu.be/3_6KJYapns4).

## Features
### Architecture tuning
Scalable MLP architecture. Its implementation includes Forward and Backward propagation, training and model hyperparameters, and activation functions.
### Optimization algorithms
$5$ optimizers implemented from scratch in NumPy:
- *Gradient Descent*
- *Stochastic Gradient Descent*
- *Stochastic Gradient with Momentum*
- *AdaGrad*
- *Adam*
### Datasets management
I've taken three publicly available datasets:
- [*Gender Voice*](https://www.kaggle.com/datasets/primaryobjects/voicegender)
- [*Heart Disease*](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- [*Iris*](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

They are served through a *MongoDB* instance.

The initial idea was to have an `Upload Dataset` option as a part of the project but it is not implemented yet. Hence, those are the only datasets on which you can train your models through this API.

### PCA
Principal Component Analysis (PCA) implemented from scratch. You can pass the dataset through this preprocessing step and extract top $n$ features.

## Examples

### Train an MLP
Train an MLP with $3$ layers, each one with $3$ neurons. The chosen optimization algorithm is $Adam$ (`"optimization": "adam"`). Each layer (except for the last one) has a Sigmoid activation function (`"activation": "sigm"`). The model will be trained for $10$ $epochs$ with batch size ($batch$) of $10$. I won't delve into the other hyperparameters, most of them are linked to the optimizer.

The other settings are for the preprocessing of the 
dataset. The chosen dataset is `iris`. Before the training of the model it is split into a training set of $70\%$ and validation set of $30\%$ (`"train_size": 0.7`). PCA is enabled (`"pca": true`), therefore we select top $3$ features/components which are going to be passed to the model.

Here is the request format:

`POST` on `http://0.0.0.0:5000/architecture`

Request body:
```json
{
    "architecture": {
        "1": 3,
        "2": 3,
        "3": 3
    },
    "optimization": "adam",
    "hyperparameters": {
        "learning_rate": 0.001,
        "type_": "classification",
        "epochs": 10,
        "batch": 10,
        "random": 10,
        "activation": "sigm",
        "momentum": 0.5,
        "epsilon": 0.0001
    },
    "dataset": "iris",
    "pca": true,
    "pca_options": {
        "n_components": 3
    },
    "train_size": 0.7
}
```

### Get available datasets
`GET` on `http://0.0.0.0:5000/datasets/names`

### Run PCA on a dataset
`POST` on `http://0.0.0.0:5000/pca`

Request body:
```json
{
    "dataset_name": "iris",
    "n_components": 4,
    "features": "all"
}
```

### Download the model weights
Download the weights of the latest trained model.

`GET` on `http://0.0.0.0:5000/model`

## Deployment
The project is deployed using Docker and Docker Compose. There are two containers. One for the *MLP API* and the other one for the *MongoDB*.
```bash
docker-compose up
```
