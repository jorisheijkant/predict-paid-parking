import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('data/park_data.csv')

# These are the parameters with which we'll train the model
# So we'll take these into account and drop all others
# Take parameters here that have values for each neighbourhood, or construct a pipeline to fill in missing values
parameters = ['a_pau', 'a_m2w', 'ste_mvs', 'a_ongeh', 'a_hh_z_k', 'a_hh_m_k']

# Set up dataset
X = data[parameters]  
y = data['paid_parking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the value counts. Here you can see that the data set is pretty imbalanced unfortunately
print(data['paid_parking'].value_counts())

# Set up the model, the class_weight here is important because we have so few paid parking zones
# Up the iterations a bit as well
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X_train, y_train)

# Do some predictions and get the model scores
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Run a prediction again, and output the results to a csv file
all_neighbourhoods = data
new_predictions = model.predict(all_neighbourhoods[parameters])
all_neighbourhoods['predicted_paid_parking'] = new_predictions
all_neighbourhoods.to_csv("output.csv")

# This logs the relative importance of the features we looked at
# With this, we can see the backside of the model 
# It will tell you what it looked at mostly for its predictions
feature_importances = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print('Feature Importance:')
print(feature_importances)