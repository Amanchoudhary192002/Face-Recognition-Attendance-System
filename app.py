import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Dates
def datetoday():
    return date.today().strftime("%m_%d_%y")

def datetoday2():
    return date.today().strftime("%d-%B-%Y")

# Load Haar cascade
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directories setup
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('Name,Roll,Time,Date,attendance')

# Total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Extract faces
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces

# Identify face using model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract today's attendance
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    return df['Name'], df['Roll'], df['Time'], df['Date'], df['attendance'], len(df)

# Mark attendance
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    dates = date.today().strftime("%d-%B-%Y")
    attendance = present_absent(current_time)
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{dates},{attendance}')

def present_absent(time):
    hour, minutes, seconds = time.split(":")
    if "10" <= hour <= "11" and "00" <= minutes <= "45":
        return "Present"
    return "Absent"

# ROUTES

@app.route('/')
def home():
    names, rolls, times, dates, attendance, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates,
                           attendance=attendance, l=l, totalreg=totalreg(), datetoday2=datetoday2())

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='No trained model found. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, dates, attendance, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates,
                           attendance=attendance, l=l, totalreg=totalreg(), datetoday2=datetoday2())

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if j % 10 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                i += 1
            j += 1

        if j == 500 or cv2.waitKey(1) == 27:
            break

        cv2.imshow('Adding new User', frame)

    cap.release()
    cv2.destroyAllWindows()

    train_model()

    names, rolls, times, dates, attendance, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates,
                           attendance=attendance, l=l, totalreg=totalreg(), datetoday2=datetoday2())

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
