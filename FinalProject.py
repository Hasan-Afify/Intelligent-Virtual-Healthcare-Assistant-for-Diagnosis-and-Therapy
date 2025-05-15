import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
import os
import io
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CNN model (updated to match old code)
try:
    cnn_model = load_model('C:/Users/Hasan/Desktop/IMAGES WITHOUT WATER MARK/skin_disease_model.h5')
    logger.info("CNN model loaded successfully")
except Exception as e:
    logger.error(f"Error loading CNN model: {e}")
    st.error("خطأ في تحميل نموذج الـ CNN، تأكد من مسار الملف")

# Simulated training data for chatbot
def create_training_data():
    data = {
        'age': [25, 30, 40, 35, 28, 45, 50, 22],
        'medical_history': [
            'itching, red spots', 'nail discoloration', 'small bumps', 'no symptoms',
            'itching, circular rash', 'thick nails', 'warts on hand', 'healthy'
        ],
        'diagnosis': [
            'Tinea Ringworm', 'Nail Fungus', 'Warts Molluscum', 'Normal',
            'Tinea Ringworm', 'Nail Fungus', 'Warts Molluscum', 'Normal'
        ]
    }
    return pd.DataFrame(data)

# Train a simple classifier for chatbot
def train_chatbot():
    df = create_training_data()
    X = df[['age']].join(pd.get_dummies(df['medical_history'])).values
    y = df['diagnosis']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf, df['medical_history'].unique()

clf, medical_history_options = train_chatbot()

# Disease-specific questions (updated to match old code class names)
disease_questions = {
    'Tinea Ringworm': [
        'هل تعاني من حكة شديدة في المنطقة المصابة؟',
        'هل الطفح الجلدي دائري الشكل؟',
        'هل تنتشر البقع الحمراء بسرعة؟'
    ],
    'Nail Fungus': [
        'هل لاحظت تغير لون الظفر إلى الأصفر أو البني؟',
        'هل الظفر أصبح سميكًا أو هشًا؟',
        'هل تعاني من ألم عند الضغط على الظفر؟'
    ],
    'Warts Molluscum': [
        'هل الثآليل صغيرة وخشنة الملمس؟',
        'هل ظهرت الثآليل في أماكن مثل اليدين أو القدمين؟',
        'هل حاولت إزالة الثآليل وعادة مرة أخرى؟'
    ],
    'Normal': [
        'هل تشعر بأي أعراض غير طبيعية؟',
        'هل تعاني من أي حساسية جلدية؟',
        'هل لاحظت أي تغيرات في الجلد مؤخرًا؟'
    ]
}

# Treatment and advice (updated to match old code class names)
treatments = {
    'Tinea Ringworm': {
        'treatment': 'استخدم كريم مضاد للفطريات مثل الكلوتريمازول مرتين يوميًا لمدة 2-4 أسابيع.',
        'advice': 'حافظ على المنطقة نظيفة وجافة، وتجنب مشاركة المناشف أو الملابس.'
    },
    'Nail Fungus': {
        'treatment': 'استخدم طلاء أظافر مضاد للفطريات مثل الأمورولفين، أو استشر طبيبًا لتناول أدوية فموية.',
        'advice': 'حافظ على الأظافر قصيرة وجافة، وارتدي أحذية جيدة التهوية.'
    },
    'Warts Molluscum': {
        'treatment': 'استخدم حمض الساليسيليك الموضعي يوميًا، أو جرب العلاج بالتجميد عند طبيب الجلدية.',
        'advice': 'تجنب لمس الثآليل أو خدشها لمنع انتشارها.'
    },
    'Normal': {
        'treatment': 'لا حاجة لعلاج، حالتك طبيعية.',
        'advice': 'حافظ على نظافة الجلد واستخدم مرطبًا إذا لزم الأمر.'
    }
}

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("تحدث الآن...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language='ar-EG')
            return text
        except sr.WaitTimeoutError:
            st.error("لم يتم تسجيل صوت، حاول مرة أخرى")
            return None
        except sr.UnknownValueError:
            st.error("لم أفهم ما قلت، حاول مرة أخرى")
            return None
        except sr.RequestError:
            st.error("خطأ في الاتصال بخدمة التعرف على الصوت")
            return None

# Function to convert text to speech
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='ar', slow=False)
    tts.save(filename)
    return filename

# Function to process image (updated to match old code)
def process_image(image):
    try:
        image_np = np.array(image)
        logger.info(f"Original image shape: {image_np.shape}")
        
        # Keep as BGR (to match old code, which uses cv2.imread or camera frames)
        if len(image_np.shape) == 2:  # Grayscale image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4:  # RGBA image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        # If RGB, convert to BGR
        elif image_np.shape[2] == 3 and image_np[0, 0, 0] > image_np[0, 0, 2]:  # Heuristic to detect RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Processed image shape: {image_np.shape}")
        return image_np
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error("خطأ في معالجة الصورة، تأكد من أن الصورة صالحة")
        return None

# Function to predict disease (updated to match old code)
def predict_disease(image):
    try:
        # Resize and normalize image
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        logger.info(f"Image shape after preprocessing: {image.shape}")

        # Predict
        prediction = cnn_model.predict(image)
        logger.info(f"Prediction probabilities: {prediction}")

        # Updated class names to match old code
        labels = ['Nail Fungus', 'Normal', 'Tinea Ringworm', 'Warts Molluscum']
        predicted_label = labels[np.argmax(prediction)]
        logger.info(f"Predicted label: {predicted_label}")
        return predicted_label
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error("خطأ في التنبؤ، تأكد من نموذج الـ CNN والصورة")
        return None

# Streamlit app
def main():
    st.title("الدكتور الذكي")
    st.write("مرحبًا! أنا chatbot طبي لتشخيص الأمراض الجلدية. يمكنك التحدث معي بالصوت!")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 'collect_data'
        st.session_state.patient_data = {}
        st.session_state.diagnosis = None
        st.session_state.current_question = 0
        st.session_state.answers = []
        st.session_state.image_source = None

    # Step 1: Collect patient data
    if st.session_state.step == 'collect_data':
        st.subheader("أدخل بيانات المريض")
        
        # Name
        st.write("قل اسمك أو أدخله يدويًا")
        if st.button("تسجيل الاسم بالصوت"):
            name = recognize_speech()
            if name:
                st.session_state.patient_data['name'] = name
                st.write(f"الاسم المسجل: {name}")
        name = st.text_input("اسم المريض", value=st.session_state.patient_data.get('name', ''))

        # Age
        st.write("قل عمرك أو أدخله يدويًا")
        if st.button("تسجيل العمر بالصوت"):
            age_text = recognize_speech()
            if age_text:
                try:
                    age = int(''.join(filter(str.isdigit, age_text)))
                    st.session_state.patient_data['age'] = age
                    st.write(f"العمر المسجل: {age}")
                except ValueError:
                    st.error("لم أفهم العمر، أدخله يدويًا")
        age = st.number_input("العمر", min_value=0, max_value=120, step=1, value=st.session_state.patient_data.get('age', 0))

        # Medical History
        st.write("قل التاريخ المرضي أو أدخله يدويًا")
        if st.button("تسجيل التاريخ المرضي بالصوت"):
            history = recognize_speech()
            if history:
                st.session_state.patient_data['history'] = history
                st.write(f"التاريخ المرضي المسجل: {history}")
        history = st.text_area("التاريخ المرضي", value=st.session_state.patient_data.get('history', ''))

        if st.button("إرسال البيانات"):
            if not name or not age or not history:
                st.error("يرجى ملء جميع الحقول")
                error_audio = text_to_speech("يرجى ملء جميع الحقول", "error.mp3")
                st.audio(error_audio)
            else:
                st.session_state.patient_data = {'name': name, 'age': age, 'history': history}
                st.session_state.step = 'choose_image'
                confirm_text = "تم تسجيل البيانات، الآن اختر طريقة إدخال الصورة"
                confirm_audio = text_to_speech(confirm_text, "confirm.mp3")
                st.audio(confirm_audio)
                st.rerun()

    # Step 2: Choose image source
    elif st.session_state.step == 'choose_image':
        st.subheader("اختر طريقة إدخال الصورة")
        st.write("يمكنك رفع صورة جاهزة أو التقاط صورة بالكاميرا")
        image_source = st.radio("اختر مصدر الصورة:", ("رفع صورة", "استخدام الكاميرا"))

        if image_source == "رفع صورة":
            uploaded_file = st.file_uploader("ارفع صورة (JPG أو PNG)", type=["jpg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_np = process_image(image)
                if image_np is not None:
                    st.image(image, caption="الصورة المرفوعة", use_column_width=True)
                    st.session_state.diagnosis = predict_disease(image_np)
                    if st.session_state.diagnosis:
                        st.write(f"التشخيص: {st.session_state.diagnosis}")
                        diagnosis_text = f"التشخيص هو {st.session_state.diagnosis}"
                        diagnosis_audio = text_to_speech(diagnosis_text, "diagnosis.mp3")
                        st.audio(diagnosis_audio)
                        st.session_state.step = 'ask_questions'
                        st.rerun()

        elif image_source == "استخدام الكاميرا":
            img_file = st.camera_input("التقاط صورة")
            if img_file:
                image = Image.open(img_file)
                image_np = process_image(image)
                if image_np is not None:
                    st.session_state.diagnosis = predict_disease(image_np)
                    if st.session_state.diagnosis:
                        st.write(f"التشخيص: {st.session_state.diagnosis}")
                        diagnosis_text = f"التشخيص هو {st.session_state.diagnosis}"
                        diagnosis_audio = text_to_speech(diagnosis_text, "diagnosis.mp3")
                        st.audio(diagnosis_audio)
                        st.session_state.step = 'ask_questions'
                        st.rerun()

    # Step 3: Ask disease-specific questions
    elif st.session_state.step == 'ask_questions':
        st.subheader("أسئلة إضافية")
        if st.session_state.current_question < 3:
            question = disease_questions[st.session_state.diagnosis][st.session_state.current_question]
            st.write(f"السؤال {st.session_state.current_question + 1}: {question}")
            question_audio = text_to_speech(question, f"question_{st.session_state.current_question}.mp3")
            st.audio(question_audio)

            if st.button("تسجيل الإجابة بالصوت"):
                answer = recognize_speech()
                if answer:
                    st.session_state.answers.append(answer)
                    st.write(f"الإجابة المسجلة: {answer}")
                    st.session_state.current_question += 1
                    st.rerun()

            answer = st.text_input("إجابتك", key=f"q{st.session_state.current_question}")
            if st.button("إرسال الإجابة"):
                if not answer:
                    st.error("يرجى إدخال إجابة")
                    error_audio = text_to_speech("يرجى إدخال إجابة", "error_answer.mp3")
                    st.audio(error_audio)
                else:
                    st.session_state.answers.append(answer)
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            st.session_state.step = 'show_treatment'
            st.rerun()

    # Step 4: Show treatment and advice
    elif st.session_state.step == 'show_treatment':
        st.subheader("النتيجة النهائية")
        treatment_info = treatments[st.session_state.diagnosis]
        result_text = (
            f"التشخيص النهائي: {st.session_state.diagnosis}\n"
            f"العلاج: {treatment_info['treatment']}\n"
            f"نصائح: {treatment_info['advice']}"
        )
        st.write(f"**التشخيص النهائي**: {st.session_state.diagnosis}")
        st.write(f"**العلاج**: {treatment_info['treatment']}")
        st.write(f"**نصائح**: {treatment_info['advice']}")
        result_audio = text_to_speech(result_text, "result.mp3")
        st.audio(result_audio)

        if st.button("إعادة البدء"):
            st.session_state.step = 'collect_data'
            st.session_state.patient_data = {}
            st.session_state.diagnosis = None
            st.session_state.current_question = 0
            st.session_state.answers = []
            st.session_state.image_source = None
            st.rerun()

if __name__ == "__main__":
    main()