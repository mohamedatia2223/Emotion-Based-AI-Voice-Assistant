import cv2
import mediapipe as mp
import numpy as np

# تهيئة Mediapipe لتتبع اليد
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# إعدادات النافذة
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# متغيرات التحكم
prev_x, prev_y = None, None
drawing = False
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # ألوان قوية وواضحة
color_index = 0  # اللون الافتراضي
letter_completed = False

# قائمة لحفظ الخطوط المرسومة
lines = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # عكس الصورة لظهورها بشكل طبيعي
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # استخراج إحداثيات إصبع السبابة والإبهام
            index_finger = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            
            h, w, _ = frame.shape
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            
            # حساب المسافة بين السبابة والإبهام لتحديد انتهاء الحرف
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            distance = np.linalg.norm(np.array([x, y]) - np.array([thumb_x, thumb_y]))
            
            # التحقق مما إذا كانت المسافة بين السبابة والإبهام صغيرة
            if distance < 30:  # إذا كانت المسافة صغيرة، يتم إنهاء الحرف
                letter_completed = True
            else:
                letter_completed = False
                
            # التحقق إذا كان المستخدم يرسم بشرط أن اليد ليست مغلقة
            if prev_x is not None and prev_y is not None and not letter_completed:
                cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], 5)
                lines.append(((prev_x, prev_y), (x, y), colors[color_index]))  # حفظ الخط المرسوم
            
            prev_x, prev_y = x, y
            drawing = True  # تم اكتشاف اليد

            # رسم نقاط اليد
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # دمج الطبقات بدون شفافية
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # عرض الإطارات
    cv2.imshow("Air Writing", frame)
    
    # مفاتيح التحكم
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # لمسح الشاشة
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        lines.clear()
    elif key == ord('n'):  # تغيير اللون
        color_index = (color_index + 1) % len(colors)
    elif key == ord('z') and lines:  # التراجع عن آخر خط
        lines.pop()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        for line in lines:
            cv2.line(canvas, line[0], line[1], line[2], 5)

cap.release()
cv2.destroyAllWindows()
