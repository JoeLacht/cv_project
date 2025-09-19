import streamlit as st
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

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

# --- обработка изображений ---
if images:
    st.divider()
    st.subheader("⚙️ Результаты обработки")

for idx, image in enumerate(images):
    with st.container():
        st.markdown(f"### 🖼 Изображение {idx+1}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Оригинал", use_container_width=True)

        # Конвертация для OpenCV
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Детекция кораблей
        results = model2(img_cv)
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # Рисуем прямоугольник вокруг корабля
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Конвертация обратно в PIL
        result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)

        with col2:
            st.image(result_pil, caption="Найденные корабли", use_container_width=True)

        st.download_button(
            f"📥 Скачать результат {idx+1}",
            data=BytesIO(cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))[1].tobytes()),
            file_name=f"ships_{idx+1}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    st.divider()