# Predict paid parking

In this repository, we take a look at paid parking in the Groot Waterland area in The Netherlands. Using existing data from the national statistics bureau (CBS) we try and predict which areas will get paid parking next.

## Prerequisites

In order to run this code, you'll need Python and pip. Install the needed libraries (preferably in a separate environment like conda or venv) with `pip install -r requirements.txt`.

In order to run this script, navigate with your terminal to this folder and run `python predict_parking.py`. You should get a lot of info logged to your terminal and a csv output.

## N.B.

- Most info about the script and how it works is in the
- Spoiler alert: the model will probably assign great importance to urbanization. Please note that this indicator is reversed in the data. [A low number means high urbanization](https://www.cbs.nl/nl-nl/onze-diensten/methoden/begrippen/stedelijkheid--van-een-gebied--).
