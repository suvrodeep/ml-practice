import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Function to encode text to get embeddings
def get_embeddings(texts, batch_size):
    all_embeddings = []
    print(f"Total number of records: {len(texts)}")
    print(f"Num batches: {(len(texts) // batch_size) + 1}")

    for start_index in tqdm(range(0, len(texts), batch_size)):
        # Print batch information
        batch_texts = texts[start_index:start_index + batch_size]

        # Generate tokens and move input tensors to GPU
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract the embeddings. No grad because this is not a learning task
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the last hidden stated and pool them into a mean vector calculated across the sequence length dimension
        # This will reduce the output vector from [batch_size, sequence_length, hidden_layer_size]
        # to [batch_size, hidden_layer_size] thereby generating the embeddings for all the sequences in the batch
        last_hidden_states = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_states, dim=1).cpu().tolist()

        # Append to the embeddings list
        all_embeddings.extend(embeddings)

    return all_embeddings


def train_model(data, labels):
    if torch.cuda.is_available():
        boost_device = "cuda"
    else:
        boost_device = "cpu"

    # Initialize the XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                                device=boost_device,
                                random_state=3137)

    # Define hyperparameters and values to tune
    param_grid = {
        'max_depth': [4, 5, 6, 7, 8],
        'eta': np.arange(0.05, 0.3, 0.05)
    }

    print(f"Number of rows in training data: {len(data)}")

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring="roc_auc",
                               cv=5, verbose=3)
    grid_search.fit(data, labels)

    # Get the best hyperparameters
    best_max_depth = grid_search.best_params_['max_depth']
    best_eta = grid_search.best_params_['eta']

    final_xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                                      max_depth=best_max_depth,
                                      eta=best_eta,
                                      device=boost_device,
                                      random_state=3137)
    final_xgb_clf.fit(data, labels)

    return final_xgb_clf


def main():
    # Load training data and generate embeddings
    df = pd.read_csv("./data/twitter_sentiment_analysis/train.csv")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != "label"], df["label"], test_size=0.3,
                                                        stratify=df["label"])

    # Get embeddings for the training and test set
    train_embeddings = get_embeddings(texts=X_train["tweet"].tolist(), batch_size=256)
    train_embeddings_df = pd.DataFrame(train_embeddings)

    test_embeddings = get_embeddings(texts=X_test["tweet"].tolist(), batch_size=256)
    test_embeddings_df = pd.DataFrame(test_embeddings)

    # Train model
    xgb_model = train_model(data=train_embeddings_df, labels=y_train)

    # Predict from model
    y_pred = xgb_model.predict(test_embeddings_df)

    # Evaluate model
    print(f"Classification report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    main()

