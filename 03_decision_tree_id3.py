"""
Experiment 3: Decision Tree (ID3 Algorithm)
Demonstrate the working of the decision tree based ID3 algorithm using an appropriate dataset
and classify a new sample.
"""

import math
import pandas as pd


def get_play_tennis_data():
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                    "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                        "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                     "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                 "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                       "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }
    return pd.DataFrame(data)


def entropy(target_col):
    counts = target_col.value_counts()
    total = len(target_col)
    return -sum((count / total) * math.log2(count / total) for count in counts)


def info_gain(df, split_attr, target_name="PlayTennis"):
    total_entropy = entropy(df[target_name])
    total = len(df)
    weighted_entropy = sum(
        (len(subset) / total) * entropy(subset[target_name])
        for _, subset in df.groupby(split_attr)
    )
    return total_entropy - weighted_entropy


def id3(df, features, target_name="PlayTennis"):
    if len(df[target_name].unique()) == 1:
        return df[target_name].iloc[0]
    if not features:
        return df[target_name].mode()[0]

    gains = {feat: info_gain(df, feat, target_name) for feat in features}
    best_feat = max(gains, key=gains.get)
    tree = {best_feat: {}}

    remaining_features = [f for f in features if f != best_feat]
    for val, subset in df.groupby(best_feat):
        tree[best_feat][val] = id3(subset, remaining_features, target_name)

    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    val = sample.get(root)
    subtree = tree[root].get(val)
    if subtree is None:
        return "Unknown"
    return predict(subtree, sample)


if __name__ == "__main__":
    df = get_play_tennis_data()
    features = [col for col in df.columns if col != "PlayTennis"]
    decision_tree = id3(df, features)
    print("Constructed ID3 Decision Tree:\n", decision_tree)

    new_sample = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
    prediction = predict(decision_tree, new_sample)
    print(f"\nNew Sample: {new_sample}")
    print(f"Predicted Class (PlayTennis): {prediction}")
