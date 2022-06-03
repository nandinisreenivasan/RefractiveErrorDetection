from flask import Flask,render_template,request,Response,jsonify
from DistanceCalculator import Calci
from measure import measure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import cv2
import pandas as pd
import pickle

app = Flask(__name__)
camera = cv2.VideoCapture(0)

lst = [21,	77,	48,	48,	48,0,0,	48,	48,	48,	48,	0]

model = pickle.load(open('model.pkl', 'rb'))
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/newyork', methods = ['POST', 'GET'])
def newyork_page():
   if request.method == 'POST':
      result = request.form
      lst.append(result['ag'])
      return render_template("login.html",result = result)

@app.route('/stanford',methods = ['POST', 'GET'])
def stanford_page():
    return render_template("change.html")
    
@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    lt=[21,	77,	48,	48,	48,0,0,	48,	48,	48,	48,	0]
        
    dataset = pd.read_csv('C:/Users/nandi/final_project/test/Distance/ds.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    ddd=(classifier.predict(sc.transform([lt])))
    return render_template('working.html',prediction=(ddd))

if __name__ == '__main__':
   app.run(debug = True)
