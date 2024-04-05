# NBA Player Career Duration Prediction Pipeline

This folder contains the code for the NBA player career duration prediction pipeline, wrapped in a FastAPI application.

## Setup

To create the virtual environment, run this command: `python -m venv env`

To activate it, run this command: `source env/bin/activate`

Now, install the dependencies using this command: `pip install -r requirements.txt`

## Train and Test Model (XGBoost)

To train and test the model and save the model as well as the scaler used to scale the data, run this command: `python model.py`

The `main` function used in `model.py` takes the following arguments:

- `enable_plot`: A boolean indicating whether to enable plotting during model training and testing.

- `n_components_pca`: An integer specifying the number of components for Principal Component Analysis (PCA) when performing dimensionality reduction.
  
- `clf_name`: A string specifying the name to be used for saving the trained model. The model will be saved in pickl format if this argument is specified.

- `scaler_name`: A string specifying the name to be used for saving the scaler used to scale the data. The scaler will be saved if this argument is specified.

After running `model.py`, the model will be saved in pickl format if `clf_name` is specified, as well as the scaler if `scaler_name` is specified.

The same scoring function `score_classifier` given by default in the `test.py` at the beginning is kept to test the model, except we increased the number of Folds for cross-validation from 3 to 5 for better scoring.


# NBA Player Career Duration Prediction API

This API predicts the career duration of NBA players using a pre-trained XGBoost model.
It accepts a JSON containing players' statistics and returns a JSON with predictions of whether the player will continue in the league for the next 5 years or not.
Each prediction includes the index of the player, the predicted probability, and a message to the user.

## Endpoints
1. `/`: Root endpoint providing a simple message.
2. `/predict/`: Endpoint for making predictions based on input player data.

## Usage
To launch the application, execute the following command: `python app.py`. Once the application is running, you can test the API by using the command: `python test_app.py`. If you want to modify the test input data, simply edit the `test_requests.json` file.

## Documentation

The auto-generated FastAPI documentation can be accessed here: http://127.0.0.1:8000/docs.
