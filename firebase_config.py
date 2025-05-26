import firebase_admin
from firebase_admin import credentials, db

def init_firebase():
    cred = credentials.Certificate("E:/firebase/firebase-key.json")  # path ke key JSON kamu
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://realtimevehicle-c14f0-default-rtdb.asia-southeast1.firebasedatabase.app/'  # ganti dengan URL-mu
    })
