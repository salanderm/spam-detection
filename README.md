# Spam Detection Project

This project aims to classify text messages as either "spam" or "ham" (not spam) using machine learning. The dataset includes thousands of labeled messages, which are used to train and evaluate a Naive Bayes classifier.

## Project Overview

### Dataset
The dataset (`spam.csv`) contains:
- **Columns**:
  - `class`: Indicates whether the message is "ham" or "spam".
  - `message`: The content of the SMS message.
  - Additional unnamed columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`), which are not used in the analysis.
- **Rows**: 5,572 messages.
- **Examples**:
  - **Ham**: "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
  - **Spam**: "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121..."

### Notebook
The notebook (`spam.ipynb`) contains:
1. **Data Loading**: Reads and inspects the dataset using `pandas`.
2. **Text Preprocessing**:
   - Tokenization and vectorization of text using `CountVectorizer`.
3. **Model Training**:
   - A Naive Bayes classifier (`MultinomialNB`) is trained on the vectorized text data.
4. **Evaluation**:
   - The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Project Features
- **Input**: Text messages.
- **Output**: Prediction of whether the message is "spam" or "ham".
- **Machine Learning Model**: Naive Bayes classifier, suitable for text classification tasks.

## How to Run

1. **Setup**:
   - Install Python 3.x and Jupyter Notebook.
   - Install the required libraries:
     ```bash
     pip install pandas numpy scikit-learn
     ```

2. **Run the Notebook**:
   - Place `spam.csv` in the same directory as `spam.ipynb`.
   - Open and execute the notebook in Jupyter Notebook or JupyterLab.

3. **Customize**:
   - Modify the preprocessing pipeline or try alternative machine learning models.

## Requirements

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`

## Outputs

- A trained Naive Bayes classifier for spam detection.
- Evaluation metrics to assess the model's performance.
