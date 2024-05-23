import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Check for consistency in data shapes
lengths = [len(d) for d in data]
max_length = max(lengths)

# Pad the shorter lists
data_padded = []
for d in data:
    if len(d) < max_length:
        d = d + [0] * (max_length - len(d))
    data_padded.append(d)

# Convert to numpy arrays
data = np.asarray(data_padded)
labels = np.asarray(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
