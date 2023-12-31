from flask import Flask,request,app,jsonify,url_for,render_template
import pickle
import numpy
import pandas
app=Flask(__name__)
cl_model=pickle.load(open("telecom_model.pkl", 'rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(numpy.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(numpy.array(list(data.values())).reshape(1,-1))
    output=cl_model.predict(new_data)
    return jsonify(output[0])
@app.route('/predict',methods=['POST'])
def predict():
    data_list= request.form.values()
    data=list(data_list)
    new_data=scaler.transform(numpy.array(data).reshape(1,-1))
    print(new_data)
    output=cl_model.predict(new_data)[0]
    return render_template("home.html",predict_text="the predicted coustomer status is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)