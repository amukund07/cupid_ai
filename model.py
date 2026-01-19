import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("crush_interest_full_dataset.csv")


# Splitting data set
X = df.drop(columns=["target"])
Y = df["target"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)



model= RandomForestRegressor(max_depth=10, random_state=42)
model.fit(X_train,Y_train)


train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

def predict_interest(single_input):
    input_df = pd.DataFrame([single_input])

    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Correct column order
    input_df = input_df[X_train.columns]

    score = model.predict(input_df)[0]

    return score

import joblib
joblib.dump(model, "model.pkl")


if __name__ == "__main__":

    single_input = {
        "initiates_conversation": 7,
        "engagement": 8,
        "quick_responses": 6,
        "eye_contact": 7,
        "joke_responses": 6,
        "nervous": 2,
        "stays_near_you": 8,
        "helps_you": 7,
        "smiles": 9,
        "takes_you_out": 6
    }

    score = predict_interest(single_input)
    print(f"Predicted Interest Score: {score:.2f}")
