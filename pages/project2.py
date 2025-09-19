import streamlit as st
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os

# Загружаем модель кораблей
model2 = YOLO("models/best_ship.pt")  # твой вес модели

st.set_page_config(page_title="Ship Detection App", page_icon="🚢", layout="centered")

st.title("🚢 Ship Detection App")
st.markdown(
    """
    ### 🔬 Поиск кораблей на аэроснимках
    Загрузите одно или несколько изображений или вставьте ссылки, и модель покажет найденные корабли на изображениях.
    """
)

# --- выбор источника ---
st.divider()
st.subheader("📥 Загрузка изображений")

option = st.radio("Выберите источник:", ("📂 Файлы", "🌐 Ссылки (URL)"), horizontal=True)

images = []

# ------------------ Загрузка файлов ------------------
if option == "📂 Файлы":
    uploaded_files = st.file_uploader(
        "Выберите изображение(я)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            try:
                img = Image.open(BytesIO(f.read())).convert("RGB")
                images.append(img)
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла {f.name}: {e}")

# ------------------ Загрузка по URL ------------------
elif option == "🌐 Ссылки (URL)":
    urls = st.text_area("Вставьте ссылки (по одной на строке)", height=100,
                        placeholder="https://example.com/image1.jpg")
    if urls:
        for url in urls.splitlines():
            url = url.strip()
            if url:
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    images.append(img)
                except Exception as e:
                    st.error(f"❌ Ошибка загрузки: {url} ({e})")

# ------------------ Обработка изображений ------------------
if images:
    st.divider()
    st.subheader("⚙️ Результаты обработки")

    for idx, image in enumerate(images):
        with st.container():
            st.markdown(f"### 🖼 Изображение {idx+1}")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Оригинал", width="stretch")

            # Конвертация для OpenCV
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            # Детекция кораблей
            results = model2(img_cv)
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Конвертация обратно в PIL
            result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img)

            with col2:
                st.image(result_pil, caption="Найденные корабли", width="stretch")

            # Кнопка для скачивания
            st.download_button(
                f"📥 Скачать результат {idx+1}",
                data=BytesIO(cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))[1].tobytes()),
                file_name=f"ships_{idx+1}.jpg",
                mime="image/jpeg",
                key=f"download_{idx+1}",
                help="Скачать обработанное изображение с прямоугольниками вокруг кораблей"
            )

        st.divider()

# --- Секция обучения модели (корабли) ---
st.subheader("🚢 Обучение модели (Корабли)")
with st.expander("Посмотреть детали обучения"):
    st.markdown("""
    Модель обучалась суммарно 50 эпох (с учётом всех перезапусков).  
    Итоговые метрики:
    
    - Precision: 0.582 — предсказания не всегда точные, но стремятся быть такими  
    - Recall: 0.489 — пропускает часть объектов  
    - mAP50: 0.488 — результат средний  
    - mAP50-95: 0.297 — видно, что модель пока «сыровата», но работает
    
    🔎 Итог: модель обучилась базово, но требует доработки (например, больше данных или аугментаций)
    """)
    
        # График обучения
    if os.path.exists("images/results.png"):
        st.image("images/results.png", caption="График обучения модели", use_container_width=True)
    else:
        st.info("График обучения пока отсутствует. Поместите файл 'images/results.png' в папку проекта.")

    # --- Коллаж с тестами ---
    if os.path.exists("images/collage2.png"):
        st.image("images/collage2.png", caption="Коллаж тестов", use_container_width=True)
    else:
        st.info("Коллаж тестов отсутствует. Поместите файл 'images/collage2.png' в папку images.")