from flask import Flask,render_template,request
import joblib

#intiliaze the app
app = Flask(__name__)
model = joblib.load(r'C:\Users\RAJAT DEVARAKONDA\Desktop\Data_Science\PROJECTS DEPLOYMENTS\telecome_chrun\flask_model\Churn.pkl')
print('[INFO] model loaded')

#__name__ refers that this file main file in the module

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict' , methods = ['post'])
def predict():
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    tenure = request.form.get('tenure')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    PaperlessBilling = request.form.get('PaperlessBilling')
    MonthlyCharges = request.form.get('MonthlyCharges')
    TotalCharges = request.form.get('TotalCharges')
    InternetService_DSL = request.form.get('InternetService_DSL')
    InternetService_Fiber_optic = request.form.get('InternetService_Fiber_optic')
    Contract_One_year = request.form.get('Contract_One_year')
    Contract_Two_year= request.form.get('Contract_Two_year')
    PaymentMethod_Credit_card = request.form.get('PaymentMethod_Credit_card')
    PaymentMethod_Electronic= request.form.get('PaymentMethod_Electronic')
    PaymentMethod_Mailed = request.form.get('PaymentMethod_Mailed')
    print(SeniorCitizen, Partner,Dependents,tenure,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,PaperlessBilling,MonthlyCharges,TotalCharges,InternetService_DSL,InternetService_Fiber_optic,Contract_One_year,Contract_Two_year,PaymentMethod_Credit_card,PaymentMethod_Electronic,PaymentMethod_Mailed)
    output = model.predict([[SeniorCitizen, Partner,Dependents,tenure,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,PaperlessBilling,MonthlyCharges,TotalCharges,InternetService_DSL,InternetService_Fiber_optic,Contract_One_year,Contract_Two_year,PaymentMethod_Credit_card,PaymentMethod_Electronic,PaymentMethod_Mailed]])
    if output[0]==1:
        print('Cutomer Will Chrun')
        result = 'Chrun'
    else:
        print('Cutomer Will Not Chrun')
        result = 'Not Chrun'
    return render_template('predict.html',predict=f'Customer Will {result}')


# run the app
app.run(debug=True)


