
def naive_essay_length_model(train, test, expected_outcomes):
    predictions = []
    for row in sorted(test.values, key=lambda x: x[0]):
        predictions.append(1 if 1600 < row[-1] else 0)
    return predictions
