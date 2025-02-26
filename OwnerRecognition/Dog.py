import cv2
from deepface import DeepFace
import pyodbc
import os

# الاتصال بقاعدة البيانات
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=NONAME;'  # استبدله باسم السيرفر الحقيقي
    'DATABASE=Person_Details;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# تحميل بيانات أصحاب الكلب من قاعدة البيانات
cursor.execute("SELECT name, age, image_path FROM owners")
known_faces = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}  # {name: (age, image_path)}

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        temp_img_path = "temp_frame.jpg"
        cv2.imwrite(temp_img_path, frame)  # حفظ الصورة المؤقتة من الكاميرا

        recognized = False
        recognized_name = ""
        recognized_age = ""

        for name, (age, img_path) in known_faces.items():
            if os.path.exists(img_path):  # التأكد من أن الصورة موجودة
                result = DeepFace.verify(temp_img_path, img_path, model_name='VGG-Face', enforce_detection=False)
                if result['verified']:
                    recognized_name = name
                    recognized_age = age
                    recognized = True
                    break  # بمجرد التعرف على الشخص نخرج من الحلقة

        # عرض الاسم والعمر على الشاشة في حال التعرف على الشخص
        if recognized:
            text = f"{recognized_name}, Age: {recognized_age}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        print("خطأ في التعرف على الوجه:", e)

    cv2.imshow("Dog Owner Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
