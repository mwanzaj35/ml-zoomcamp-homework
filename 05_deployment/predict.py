import pickle

# we would like to make predictions on datapoint below
datapoint = {
    'gender': 'male',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 129.85
}


# load saved model
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


# pipeline does transformation
churn = pipeline.predict_proba(datapoint)[0, 1]

if churn >= 0.5:
    print('Send email with promo')
else:
    print('Do nothing')

