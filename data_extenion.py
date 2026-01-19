import pandas as pd
import numpy as np

df = pd.read_csv("crush_interest_dataset.csv")


def augment_data(df, n_samples=500, noise_level=0.8):
    augmented_rows = []

    feature_cols = df.columns.drop("target")

    for _ in range(n_samples):
        # pick a random existing row
        base = df.sample(1).iloc[0]

        new_row = {}

        for col in feature_cols:
            noise = np.random.normal(0, noise_level)
            value = base[col] + noise
            value = int(round(np.clip(value, 0, 9)))
            new_row[col] = value

        # target with softer noise
        target_noise = np.random.normal(0, 5)
        target = int(round(np.clip(base["target"] + target_noise, 0, 100)))
        new_row["target"] = target

        augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows)

augmented_df = augment_data(df, n_samples=1000)

full_df = pd.concat([df, augmented_df], ignore_index=True)
full_df.to_csv("crush_interest_full_dataset.csv", index=False)

print(full_df.shape)
print(full_df)