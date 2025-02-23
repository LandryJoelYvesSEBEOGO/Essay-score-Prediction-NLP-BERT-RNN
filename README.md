# Essay Score Prediction with Deep Learning

## Project Description
This project aims to develop a deep learning model capable of predicting essay scores based on teacher annotations. It leverages Natural Language Processing (NLP) techniques to extract text features and trains a regression model using a Bidirectional Gated Recurrent Unit (Bi-GRU) network.
![model output](chemin/vers/image.png)

## Key Features
- **Text Preprocessing**: Essay cleaning, stopword removal, and lemmatization.
- **Feature Extraction**: Utilizing BERT to generate sentence embeddings.
- **Modeling**: Training a Bi-GRU model for score prediction.
- **Validation**: Evaluating the model using 5-fold cross-validation.
- **Visualization**: Comparing predicted scores with actual scores.

## Installation

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy torch transformers tensorflow scikit-learn nltk seaborn matplotlib tqdm
```

### Configuration
In the notebook file, make sure that necessary resources, such as BERT model weights, are properly downloaded.

## Usage
1. **Load Data**: The CSV file containing essays is loaded and preprocessed.
2. **Extract BERT Embeddings**: Texts are transformed into vectors using BERT.
3. **Train the Model**: A Bi-GRU model is built and trained on the embeddings.
4. **Predict and Evaluate**: The model predicts essay scores, and performance is evaluated using metrics like Mean Absolute Error (MAE) and Cohen's Kappa score.

## Code Structure
- `clean_essays()`: Cleans and tokenizes essays.
- `get_bert_embeddings()`: Extracts embeddings using BERT.
- `getFeatureVecs()`: Converts embeddings into matrices for the model.
- `get_model()`: Defines the Bi-GRU model.
- `train_and_evaluate()`: Performs 5-fold cross-validation.

## Results & Analysis
The model's performance is evaluated by comparing predicted scores with actual teacher-assigned scores. Results are visualized using comparison plots.

## Possible Improvements
- Experimenting with other NLP models (GPT, RoBERTa, T5).
- Hyperparameter tuning for better accuracy.
- Expanding the dataset for improved generalization.
