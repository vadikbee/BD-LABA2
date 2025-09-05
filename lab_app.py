import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
import base64
import io  # <-- Добавлен необходимый импорт

# --- Конфигурация страницы и тема "Ночная сакура" ---

st.set_page_config(
    page_title="Предсказание цен на квартиры",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Функция для кодирования изображения в base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Функция для установки фона
def set_background(jpg_file):
    bin_str = get_base64(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Попробуем установить фон, если файл существует
try:
    set_background('sakura.jpg')
except FileNotFoundError:
    st.warning("Файл 'sakura.jpg' не найден. Фон не будет установлен.")

# Кастомный CSS для улучшения темы
st.markdown("""
<style>
    /* Основной текст */
    .stApp {
        color: #FFFFFF;
    }
    /* Заголовки */
    h1, h2, h3 {
        color: #FFC0CB; /* Розовый цвет сакуры */
    }
    /* Боковая панель */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.6);
        border-right: 2px solid #FFC0CB;
    }
    /* Виджеты в боковой панели */
    .stSelectbox, .stNumberInput, .stButton {
        color: #FFFFFF;
    }
    /* Кнопки */
    .stButton>button {
        background-color: #FF69B4;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #FFC0CB;
    }
    .stButton>button:hover {
        background-color: #FFC0CB;
        color: #000000;
        border: 1px solid #FF69B4;
    }
    /* Метрики */
    .stMetric {
        background-color: rgba(40, 40, 40, 0.7);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #FF69B4;
    }
    /* Рамка для графиков */
    .stPlotlyChart {
        border: 1px solid #FFC0CB;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Настройка стиля для графиков Matplotlib
matplotlib.rc('axes', facecolor='black', edgecolor='pink', labelcolor='white', titlecolor='pink')
matplotlib.rc('xtick', color='white')
matplotlib.rc('ytick', color='white')
matplotlib.rc('figure', facecolor='black', edgecolor='pink')
matplotlib.rc('legend', facecolor='black', edgecolor='pink', labelcolor='white')


# --- Функции для загрузки и обработки данных (с кэшированием) ---

@st.cache_data
def load_data(file_path):
    """Загружает данные из CSV файла."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Файл не найден: {file_path}")
        return None


@st.cache_data
def preprocess_data(df, is_train=True):
    """Выполняет предобработку данных: очистка, кодирование."""
    df_processed = df.copy()

    # Удаление ненужных столбцов, если они есть
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)

    # Заполнение пропусков (простая стратегия)
    for col in ['Living area', 'Kitchen area', 'Floor']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    df_processed = df_processed.dropna()

    # Факторизация колонок с высокой кардинальностью
    for col in ['Metro station', 'Region']:
        if col in df_processed.columns:
            df_processed[col] = pd.factorize(df_processed[col])[0]

    # One-Hot Encoding для колонок с низкой кардинальностью
    df_processed = pd.get_dummies(df_processed, columns=['Apartment type', 'Renovation'], dummy_na=False)

    # Для валидационного набора убедимся, что все колонки из трейна присутствуют
    if not is_train:
        # Убедимся, что train_columns сохранены в session_state
        if 'train_columns' in st.session_state:
            train_cols = st.session_state.get('train_columns', [])
            # Уберем целевую переменную, если она случайно попала в список колонок
            if 'Price' in train_cols:
                train_cols.remove('Price')

            current_cols = df_processed.columns.tolist()
            missing_cols = set(train_cols) - set(current_cols)
            for c in missing_cols:
                df_processed[c] = 0

            extra_cols = set(current_cols) - set(train_cols)
            df_processed = df_processed.drop(columns=list(extra_cols))

            df_processed = df_processed[train_cols]
        else:
            st.error(
                "Колонки для обучения не были сохранены. Пожалуйста, сначала перейдите в раздел 'Обработка и обучение'.")
            return None

    return df_processed


@st.cache_resource
def train_models(X_train, y_train):
    """Обучает модели регрессии."""
    models = {}

    with st.spinner('Обучение Random Forest...'):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf

    with st.spinner('Обучение XGBoost...'):
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_reg.fit(X_train, y_train)
        models['XGBoost'] = xgb_reg

    return models


# --- Основная часть приложения ---

st.title('🌸 Предсказание стоимости квартир в Москве')
st.write('Интерактивная лабораторная работа на основе методички.')

# Загрузка данных
df_train_raw = load_data('MSK_Price_train.csv')
df_valid_raw = load_data('MSK_Price_valid.csv')

if df_train_raw is not None and df_valid_raw is not None:

    st.sidebar.header("Панель управления")
    app_mode = st.sidebar.selectbox(
        "Выберите раздел:",
        ["Обзор данных", "Обработка и обучение", "Прогнозирование цены"]
    )

    # --- РАЗДЕЛ 1: Обзор данных ---
    if app_mode == "Обзор данных":
        st.header("1. Исходные данные для обучения")
        st.write("Первые 10 строк тренировочного набора данных:")
        st.dataframe(df_train_raw.head(10))
        st.write(f"Размерность данных: {df_train_raw.shape[0]} строк, {df_train_raw.shape[1]} столбцов.")

        st.subheader("Статистическое описание числовых признаков:")
        st.write(df_train_raw.describe())

        st.subheader("Информация о типах данных и пропусках:")

        # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
        # Вместо отображения статичного файла, динамически выводим информацию о DataFrame
        buffer = io.StringIO()
        df_train_raw.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)


    # --- РАЗДЕЛ 2: Обработка и обучение ---
    elif app_mode == "Обработка и обучение":
        st.header("2. Предобработка данных и обучение моделей")

        st.subheader("2.1. Очистка от выбросов")
        st.write("Как и в методичке, проанализируем распределение цен и удалим выбросы.")

        fig, ax = plt.subplots()
        df_train_raw['Price'].hist(bins=100, ax=ax)
        ax.set_title("Распределение цен до очистки")
        ax.set_xlabel("Цена")
        ax.set_ylabel("Частота")
        st.pyplot(fig)

        price_range = st.slider(
            'Выберите диапазон цен для фильтрации (в млн. руб.)',
            0.0, 100.0, (2.0, 50.0)
        )
        min_price, max_price = price_range[0] * 1_000_000, price_range[1] * 1_000_000

        df_train_filtered = df_train_raw[(df_train_raw['Price'] > min_price) & (df_train_raw['Price'] < max_price)]
        st.write(f"После фильтрации осталось {df_train_filtered.shape[0]} строк (из {df_train_raw.shape[0]}).")

        fig, ax = plt.subplots()
        df_train_filtered['Price'].hist(bins=100, ax=ax)
        ax.set_title("Распределение цен после очистки")
        ax.set_xlabel("Цена")
        ax.set_ylabel("Частота")
        st.pyplot(fig)

        st.subheader("2.2. Предобработка и кодирование")
        # Создаем копию для предобработки, чтобы не изменять df_train_filtered
        df_train_to_process = df_train_filtered.drop('Price', axis=1)
        df_processed = preprocess_data(df_train_to_process, is_train=True)

        # Объединяем обработанные признаки с целевой переменной
        df_processed['Price'] = df_train_filtered['Price']

        st.write("Данные после обработки (заполнение пропусков, факторизация и One-Hot кодирование):")
        st.dataframe(df_processed.head())

        # Разделение данных
        X = df_processed.drop('Price', axis=1)
        y = df_processed['Price']

        # Сохраняем названия колонок для валидационного набора
        st.session_state['train_columns'] = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("2.3. Обучение моделей и оценка")
        models = train_models(X_train, y_train)
        st.session_state['models'] = models  # Сохраняем модели для других разделов
        st.session_state['data_filtered'] = True  # Флаг, что данные были обработаны

        st.write("Оценка моделей на тестовой выборке (Mean Absolute Error):")

        col1, col2 = st.columns(2)

        for name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            if name == 'Random Forest':
                col1.metric(label=name, value=f"{mae:,.0f} руб.")
            else:
                col2.metric(label=name, value=f"{mae:,.0f} руб.")

        best_model_name = "Random Forest"
        best_model = models[best_model_name]

        st.subheader(f"2.4. Анализ лучшей модели: {best_model_name}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, best_model.predict(X_test), alpha=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        ax.set_xlabel('Настоящая цена')
        ax.set_ylabel('Предсказанная цена')
        ax.set_title('Соответствие предсказаний и реальных цен')
        st.pyplot(fig)

        st.subheader("2.5. Итоговая оценка на валидационных данных")
        df_valid_filtered = df_valid_raw[(df_valid_raw['Price'] > min_price) & (df_valid_raw['Price'] < max_price)]

        y_valid = df_valid_filtered['Price']
        X_valid_raw = df_valid_filtered.drop('Price', axis=1)
        X_valid = preprocess_data(X_valid_raw, is_train=False)

        if X_valid is not None:
            valid_pred = best_model.predict(X_valid)
            valid_mae = mean_absolute_error(y_valid, valid_pred)
            st.metric(label=f"MAE на валидационном наборе ({best_model_name})", value=f"{valid_mae:,.0f} руб.")
            st.info("Это финальная оценка качества модели на данных, которые она не видела во время обучения.")


    # --- РАЗДЕЛ 3: Прогнозирование ---
    elif app_mode == "Прогнозирование цены":
        st.header("3. Сделать прогноз")

        # Проверка, были ли модели обучены
        if 'models' not in st.session_state or not st.session_state.get('data_filtered'):
            st.warning(
                "Пожалуйста, сначала перейдите в раздел 'Обработка и обучение', чтобы подготовить данные и обучить модели.")
        else:
            st.write("Введите параметры квартиры, чтобы получить предсказание от лучшей модели (Random Forest).")

            final_model = st.session_state['models']["Random Forest"]

            # Собираем данные от пользователя
            st.sidebar.subheader("Параметры квартиры:")

            apartment_types = df_train_raw['Apartment type'].unique()
            renovations = df_train_raw['Renovation'].unique()
            regions = df_train_raw['Region'].unique()
            metro_stations = df_train_raw['Metro station'].unique()

            # Виджеты для ввода
            num_rooms = st.sidebar.slider("Количество комнат", 1, 6, 2)
            area = st.sidebar.slider("Общая площадь, м²", 20.0, 200.0, 55.0)
            living_area = st.sidebar.slider("Жилая площадь, м²", 10.0, 150.0, 35.0)
            kitchen_area = st.sidebar.slider("Площадь кухни, м²", 5.0, 50.0, 10.0)
            floor = st.sidebar.slider("Этаж", 1, 50, 5)
            num_floors = st.sidebar.slider("Всего этажей в доме", 1, 50, 12)
            minutes_to_metro = st.sidebar.slider("Минут до метро", 1, 60, 15)

            apartment_type = st.sidebar.selectbox("Тип апартаментов", apartment_types)
            renovation = st.sidebar.selectbox("Ремонт", renovations)
            region = st.sidebar.selectbox("Район", regions)
            metro_station = st.sidebar.selectbox("Станция метро", metro_stations)

            if st.sidebar.button("Предсказать стоимость"):
                # --- УЛУЧШЕНИЕ: Использование pd.concat вместо устаревшего .append() ---
                input_data = {
                    'Minutes to metro': [minutes_to_metro], 'Number of rooms': [num_rooms],
                    'Area': [area], 'Living area': [living_area], 'Kitchen area': [kitchen_area],
                    'Floor': [floor], 'Number of floors': [num_floors],
                    'Metro station': [metro_station], 'Region': [region],
                    'Apartment type': [apartment_type], 'Renovation': [renovation]
                }
                input_df = pd.DataFrame(input_data)

                # Копия исходных данных для факторизации, чтобы не изменять df_train_raw
                temp_factorize_metro = pd.concat([df_train_raw['Metro station'], input_df['Metro station']],
                                                 ignore_index=True)
                temp_factorize_region = pd.concat([df_train_raw['Region'], input_df['Region']], ignore_index=True)

                input_df['Metro station'] = pd.factorize(temp_factorize_metro)[0][-1]
                input_df['Region'] = pd.factorize(temp_factorize_region)[0][-1]

                # One-Hot Encoding
                input_df = pd.get_dummies(input_df, columns=['Apartment type', 'Renovation'])

                # Обеспечение правильного порядка и наличия всех колонок
                if 'train_columns' in st.session_state:
                    train_cols = st.session_state['train_columns']

                    # Удаляем Price, если он есть
                    if 'Price' in train_cols:
                        train_cols.remove('Price')

                    # Добавляем недостающие колонки
                    for col in set(train_cols) - set(input_df.columns):
                        input_df[col] = 0

                    # Удаляем лишние колонки и устанавливаем правильный порядок
                    input_df = input_df[train_cols]

                    prediction = final_model.predict(input_df)[0]
                    st.success(f"## Предсказанная стоимость: **{prediction:,.0f} руб.**")
                else:
                    st.error("Ошибка: не найдены колонки для обучения. Вернитесь на предыдущий шаг.")