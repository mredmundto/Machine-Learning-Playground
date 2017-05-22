import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# the Binary Classification
admissions = pd.read_csv("admissions.csv")
model = LogisticRegression()
model.fit(admissions[["gpa"]], admissions["admit"])

labels = model.predict(admissions[["gpa"]])
admissions["predicted_label"] = labels
print(admissions["predicted_label"].value_counts())
#print(admissions)

# renaming the admit to actual label
admissions["actual_label"] = admissions["admit"]
matches = admissions["predicted_label"] == admissions["actual_label"]
correct_predictions = admissions[matches]

print(correct_predictions.head())

accuracy = float(len(correct_predictions)) / len(admissions)
print(len(correct_predictions))
print(len(admissions))

print ('Accuracy of Binary Classification')
print (accuracy)


true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])

true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)
true_negatives = len(admissions[true_negative_filter])

print ('The true positive')
print (true_positives)
print ('The true negative')
print (true_negatives)

# False Positive - The model incorrectly predicted that the student would be admitted even though the student was actually rejected.
# False Negative - The model incorrectly predicted that the student would be rejected even though the student was actually admitted.


# From the previous screen
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])
false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])

sensitivity = float(true_positives) / (true_positives + false_negatives)

print ('The sensitivity')
print(sensitivity)



# From previous screens
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])
false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])
true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)
true_negatives = len(admissions[true_negative_filter])
false_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 0)
false_positives = len(admissions[false_positive_filter])
specificity = float(true_negatives) / (false_positives + true_negatives)

print ('specificity')
print(specificity)

