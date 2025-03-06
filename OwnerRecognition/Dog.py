import cv2
import face_recognition
import pyodbc
import base64
import numpy as np
from io import BytesIO
from PIL import Image


# إعدادات قاعدة البيانات
DB_CONNECTION_STRING = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=NONAME;DATABASE=Person_Details;Trusted_Connection=yes;"

# فتح الاتصال بقاعدة البيانات
conn = pyodbc.connect(DB_CONNECTION_STRING)
cursor = conn.cursor()

def load_known_faces(cursor):
    """تحميل بيانات الأشخاص المخزنة في قاعدة البيانات وتحويل الصور من Base64 إلى ترميزات وجه"""
    known_faces = {}
    
    cursor.execute("SELECT Name, Age, ImageBase64 FROM owners")
    for name, age, image_base64 in cursor.fetchall():
        
        image_data = base64.b64decode(image_base64)
        image = np.array(Image.open(BytesIO(image_data)))
        
        # استخلاص ترميز الوجه
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces[name] = (encodings[0], age)  # حفظ الاسم، الترميز، والعمر
    
    return known_faces

# تحميل الصور وترميزات الوجوه من قاعدة البيانات
known_faces = load_known_faces(cursor)

# فتح الكاميرا
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # تحويل الصورة إلى RGB للتعرف على الوجوه
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # تحديد مواقع الوجوه
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # مقارنة الوجوه الموجودة بالكاميرا مع الصور المخزنة
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([data[0] for data in known_faces.values()], face_encoding)
        face_distances = face_recognition.face_distance([data[0] for data in known_faces.values()], face_encoding)

        if True in matches:
            best_match_index = matches.index(True)
            name = list(known_faces.keys())[best_match_index]
            age = known_faces[name][1]

            print(f"تم التعرف على: {name}, العمر: {age}")

            # عرض الاسم والعمر على الشاشة
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            display_text = f"{name}, {age} Age"
            cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # عرض الفيديو
    cv2.imshow("Camera", frame)

    # الخروج عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الكاميرا وقاعدة البيانات
video_capture.release()
cursor.close()
conn.close()
cv2.destroyAllWindows()
