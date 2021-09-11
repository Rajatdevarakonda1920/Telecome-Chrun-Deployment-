import joblib
chrun_model = joblib.load('Churn.pkl')
#['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 
# 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
# 'InternetService_DSL', 'InternetService_Fiber optic','Contract_One year', 'Contract_Two year',
# 'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
output = chrun_model.predict([[1, 1,2, 50, 1,1, 0, 1, 1, 1,1, 260.95, 42119.4, 1, 1,1, 0, 1, 1, 1]])
print(output)
if output[0] == 0:
	    print('not chrun')
	
else:
	    print('Chrun')