# House_Price_Prediction

A machine learning project for house price prediction.

## Project Overview

This project includes:
- Comprehensive data exploration and visualization
- Advanced preprocessing and feature engineering
- Multiple machine learning models with hyperparameter tuning
- Model evaluation and comparison
- Feature importance analysis
- Prediction functions
- Deployment preparation

## Files

- `House_Price_Prediction.ipynb`: Main Jupyter notebook with the complete analysis
- `Model.py`: Simple script for basic data loading
- `data (1).csv`: Dataset
- `requirements.txt`: Python dependencies
- `Dockerfile`: For containerized deployment

## Setup

### Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `House_Price_Prediction.ipynb` in Jupyter or VS Code
4. Run the cells to execute the analysis

### Docker Setup
1. Build the image: `docker build -t house-price-prediction .`
2. Run the container: `docker run house-price-prediction`

This ensures the project runs with consistent versions regardless of local setup.

## Key Features

- Data preprocessing with feature engineering
- Multiple regression models (Linear, Random Forest)
- Hyperparameter optimization
- Comprehensive evaluation metrics
- Interactive visualizations
- Ready for deployment