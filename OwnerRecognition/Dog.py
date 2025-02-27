import cv2
import face_recognition
import os
import pyodbc

# إعدادات قاعدة البيانات
DB_CONNECTION_STRING = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=NONAME;DATABASE=Person_Details;Trusted_Connection=yes;"

# مسار مجلد الصور المخزنة
IMAGE_FOLDER = r"C:\\Users\\MF\\Desktop\\owners_images"

# فتح الاتصال بقاعدة البيانات
conn = pyodbc.connect(DB_CONNECTION_STRING)
cursor = conn.cursor()

def get_person_details(cursor, file_name):
    """جلب اسم وعمر الشخص من قاعدة البيانات بناءً على اسم الصورة"""
    query = "SELECT Name, Age FROM owners WHERE LOWER(image_path) LIKE LOWER(?)"
    cursor.execute(query, ('%' + file_name,))
    return cursor.fetchone()

# تحميل الصور وتحميل الترميزات (face encodings) من المجلد
known_faces = {}
for file_name in os.listdir(IMAGE_FOLDER):
    file_path = os.path.join(IMAGE_FOLDER, file_name)
    
    # التأكد من أن الملف هو صورة (ملفات .jpg أو .png مثلاً)
    if os.path.isfile(file_path) and (file_name.endswith('.jpg') or file_name.endswith('.png')):
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces[file_name] = encodings[0]  # حفظ الترميز

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
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        face_distance = face_recognition.face_distance(list(known_faces.values()), face_encoding)

        # إذا تم العثور على تطابق، نعرض البيانات
        if True in matches:
            first_match_index = matches.index(True)
            file_name = list(known_faces.keys())[first_match_index]
            person_details = get_person_details(cursor, file_name)

            if person_details:
                name, age = person_details
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
