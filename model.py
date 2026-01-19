import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("crush_interest_full_dataset.csv")



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


    input_df = input_df[X_train.columns]

    score = model.predict(input_df)[0]

    return score

import joblib
joblib.dump(model, "model.pkl")



    score = predict_interest(single_input)
    print(f"Predicted Interest Score: {score:.2f}")


