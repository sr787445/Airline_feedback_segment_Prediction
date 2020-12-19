from flask import Flask,request,  render_template
import pickle
import json

# Loading the Flask app
app = Flask(__name__,template_folder='templates')

# Loading the save data
model=pickle.load(open('model.pkl','rb'))
# Loading the saved vector model
vectoring=pickle.load(open('vectoring.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("semo.html")

# pridicting the sentiment
@app.route('/predict',methods=['POST','GET'])
def predict():
    entered_text = request.form['seg']
    int_features = [entered_text]
    int_features = vectoring.transform(int_features)
    pred = model.predict(int_features)
    if pred == True:
        result = {
            "Text" : entered_text,
            "Response" : "Positive"
        }
        with open('result.json', 'w') as json_file:
            json.dump(result, json_file)
        return render_template("index.html")
    else:
        result = {
            "Text": entered_text,
            "Response": "Negative"
        }
        # saving the file in json format
        with open('result.json', 'w') as json_file:
            json.dump(result, json_file)
        return render_template("index2.html")



if __name__ == '__main__':
    app.run(debug=True)