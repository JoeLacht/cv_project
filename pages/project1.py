import streamlit as st
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os

# Загружаем модель
model = YOLO("models/best.pt")  # твой вес модели

st.set_page_config(page_title="Face Blur App", page_icon="😎", layout="centered")

st.title("😎 Face Blur App")
st.markdown(
    """
    ### 🔒 Защита приватности
    Загрузите одно или несколько изображений или вставьте ссылки — и лица будут автоматически размыты.
    """
)

# --- выбор источника ---
st.divider()
st.subheader("📥 Загрузка изображений")

option = st.radio("Выберите источник:", ("📂 Файлы", "🌐 Ссылки (URL)"), horizontal=True)

images = []

if option == "📂 Файлы":
    uploaded_files = st.file_uploader(
        "Выберите изображение(я)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            images.append(Image.open(f).convert("RGB"))

elif option == "🌐 Ссылки (URL)":
    urls = st.text_area("Вставьте ссылки (по одной на строке)", height=100, placeholder="https://example.com/image1.jpg")
    if urls:
        for url in urls.splitlines():
            url = url.strip()
            if url:
                try:
                    response = requests.get(url)
                    images.append(Image.open(BytesIO(response.content)).convert("RGB"))
                except:
                    st.error(f"❌ Ошибка загрузки: {url}")

# --- обработка ---
if images:
    st.divider()
    st.subheader("⚙️ Результаты обработки")

for idx, image in enumerate(images):
    with st.container(border=True):
        st.markdown(f"### 🖼 Изображение {idx+1}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Оригинал", use_container_width=True)

        # Конвертация для OpenCV
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Детекция
        results = model(img_cv)
        for r in results:
            for box in r.boxes.xyxy:  # координаты лица
                x1, y1, x2, y2 = map(int, box)
                face = img_cv[y1:y2, x1:x2]
                if face.size > 0:
                    face = cv2.GaussianBlur(face, (99, 99), 30)
                    img_cv[y1:y2, x1:x2] = face

        # Конвертация обратно в PIL
        result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)

        with col2:
            st.image(result_pil, caption="С заблюренными лицами", use_container_width=True)

        st.download_button(
            f"📥 Скачать результат {idx+1}",
            data=BytesIO(cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))[1].tobytes()),
            file_name=f"blurred_{idx+1}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    st.divider()

# --- Секция обучения модели ---
st.subheader("📚 Обучение модели")
with st.expander("Посмотреть детали обучения"):
    st.markdown("""
    Модель прошла обучение в 20 эпох и показала отличные результаты:
    
    - Precision (0.893) — почти нет ложных срабатываний
    - Recall (0.803) — находит около 80% всех лиц, что очень высоко для детекции
    - mAP50 (0.871) — показатель уровня продакшн-моделей
    - mAP50-95 (0.583) — отличный результат для face detection
    
    🔎 Итог: модель обучилась очень хорошо и достигла уровня готовых pre-trained решений. Дальнейшее обучение не требуется — время тестировать её на реальных изображениях.
    """)

    # График обучения
    if os.path.exists("images/grap1.png"):
        st.image("images/grap1.png", caption="График обучения модели", use_container_width=True)
    else:
        st.info("График обучения пока отсутствует. Поместите файл 'images/grap1.png' в папку проекта.")

    # --- Коллаж с тестами ---
    if os.path.exists("images/MyСollages.png"):
        st.image("images/MyСollages.png", caption="Коллаж тестов", use_container_width=True)
    else:
        st.info("Коллаж тестов отсутствует. Поместите файл 'images/MyСollages.png' в папку images.")