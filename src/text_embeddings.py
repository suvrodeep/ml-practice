import pandas as pd
import xgboost as xgb
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from imblearn.over_sampling import ADASYN
from collections import Counter
import numpy as np
import torch
import re

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Function to encode text to get embeddings
def get_embeddings(texts: List[str], batch_size: int):
    all_embeddings = []
    print(f"Total number of records: {len(texts)}")
    print(f"Num batches: {(len(texts) // batch_size) + 1}")

    # Extract embeddings for the texts in batches
    for start_index in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start_index:start_index + batch_size]

        # Generate tokens and move input tensors to GPU
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract the embeddings. no_grad because the gradient does not need to be computed
        # since this is not a learning task
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


def train_model(data: pd.DataFrame, labels: pd.Series):
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
        'max_depth': [5, 6, 7, 8],
        'eta': np.arange(0.05, 0.3, 0.05)
    }

    print(f"\nNumber of rows in training data: {len(data)}\n")

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


def clean_sentence(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s,.!?\-#@]', '', text)

    return cleaned_text


def resample(data: pd.DataFrame, label: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    smote_model = ADASYN(random_state=3137)

    data_resampled, label_resampled = smote_model.fit_resample(X=data, y=label)

    return data_resampled, label_resampled


def main():
    # Load training data and generate embeddings
    df = pd.read_csv("./data/tweets_classification/train.csv")
    df.drop(columns=['hate_speech_count', 'offensive_language_count', 'neither_count', 'count'], inplace=True)
    label_dict = {1: 'offensive_language/hate_speech',
                  0: 'neutral'}

    # Recode the class labels
    df["class"] = df["class"].apply(lambda x: 0 if x == 2 else 1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != "class"], df["class"],
                                                        test_size=0.3,
                                                        stratify=df["class"])

    for dataset in [X_train, X_test]:
        dataset["tweet_cleaned"] = dataset["tweet"].apply(lambda x: clean_sentence(x))
        print(f'Cleaned {len(dataset["tweet_cleaned"])} records in dataset')

    # Get embeddings for the training and test set
    train_embeddings = get_embeddings(texts=X_train["tweet_cleaned"].tolist(), batch_size=256)
    train_embeddings_df = pd.DataFrame(train_embeddings)

    test_embeddings = get_embeddings(texts=X_test["tweet_cleaned"].tolist(), batch_size=256)
    test_embeddings_df = pd.DataFrame(test_embeddings)

    # Check if SMOTE needs to be applied
    class_counts = Counter(y_train)
    class_counts = [value for _, value in sorted(class_counts.items(), key=lambda item: item[1])]
    class_ratio = class_counts[0] / class_counts[1]

    if class_ratio < 0.5:
        train_df, train_labels = resample(data=train_embeddings_df, label=y_train)
        print(f"Minority class to Majority class ratio is {round(class_ratio, 2)}. SMOTE applied")
    else:
        train_df = train_embeddings_df
        train_labels = y_train
        print(f"Minority class to Majority class ratio is {round(class_ratio, 2)} which is greater than 0.5. "
              f"SMOTE not applied")

    # Train model
    xgb_model = train_model(data=train_df, labels=train_labels)

    # Predict from model
    y_pred = xgb_model.predict(test_embeddings_df)
    y_pred_labels = [label_dict[x] for x in y_pred]

    # Evaluate model
    y_test_labels = [label_dict[x] for x in y_test]
    print(f"Classification report:\n{classification_report(y_test_labels, y_pred_labels)}")


if __name__ == "__main__":
    main()

