"""
Experiment 2: Candidate-Elimination Algorithm
Implement and demonstrate the Candidate-Elimination algorithm in Python to output
a description of the set of all hypotheses consistent with the training examples.
"""

import csv
import os


def create_sample_dataset(filename="enjoysport_ce.csv"):
    data = [
        ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"],
        ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
        ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
        ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
        ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def candidate_elimination(concepts, target):
    num_attrs = len(concepts[0])
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(num_attrs)] for _ in range(num_attrs)]

    print("Initial Specific Hypothesis S0:", specific_h)
    print("Initial General Hypothesis G0:", general_h)

    for i, instance in enumerate(concepts):
        if target[i].strip().lower() == "yes":
            for x in range(num_attrs):
                if instance[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        else:
            for x in range(num_attrs):
                if instance[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

        print(f"\nStep {i + 1} ({instance} -> {target[i]}):")
        print("S:", specific_h)
        print("G:", [g for g in general_h if g != ["?"] * num_attrs])

    final_general = [g for g in general_h if g != ["?"] * num_attrs]
    return specific_h, final_general


if __name__ == "__main__":
    csv_file = "enjoysport_ce.csv"
    if not os.path.exists(csv_file):
        create_sample_dataset(csv_file)

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]

    concepts = [r[:-1] for r in rows]
    target = [r[-1] for r in rows]

    S_final, G_final = candidate_elimination(concepts, target)
    print("\nFinal Specific Hypothesis (S):", S_final)
    print("Final General Hypotheses (G):", G_final)
