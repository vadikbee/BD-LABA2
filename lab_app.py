import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder # <-- Добавлены новые импорты
import matplotlib.pyplot as plt
import matplotlib
import base64
import io

# ----- Конфигурация страницы и тема "Ночная сакура" ---
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
    page_bg_img = f"""
    <style>
    .stApp {{
    background-image: url("data:image/jpg;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    """
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
    .stApp { color: #FFFFFF; }
    /* Заголовки */
    h1, h2, h3 { color: #FFC0CB; /* Розовый цвет сакуры */ }
    /* Боковая панель */
    .css-1d391kg { background-color: rgba(0, 0, 0, 0.6); border-right: 2px solid #FFC0CB; }
    /* Виджеты в боковой панели */
    .stSelectbox, .stNumberInput, .stButton, .stSlider { color: #FFFFFF; }
    /* Кнопки */
    .stButton>button { background-color: #FF69B4; color: #FFFFFF; border-radius: 8px; border: 1px solid #FFC0CB; }
    .stButton>button:hover { background-color: #FFC0CB; color: #000000; border: 1px solid #FF69B4; }
    /* Метрики */
    .stMetric { background-color: rgba(40, 40, 40, 0.7); border-radius: 10px; padding: 10px; border: 1px solid #FF69B4;}
    /* Рамка для графиков */
    .stPlotlyChart, .stImage, .stPyPlot { border: 1px solid #FFC0CB; border-radius: 10px; }
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

# ИСПРАВЛЕНИЕ: Новая, более надежная функция предобработки
@st.cache_data
def preprocess_data_new(df):
    """Выполняет предобработку данных с использованием сохраненных кодировщиков и созданием признаков."""
    df_processed = df.copy()

    # УЛУЧШЕНИЕ 1: Создание новых признаков (Feature Engineering)
    # Используем .get() для безопасного доступа к столбцам, которых может не быть в input_df
    floor = df_processed.get('Floor')
    num_floors = df_processed.get('Number of floors')
    living_area = df_processed.get('Living area')
    area = df_processed.get('Area')
    kitchen_area = df_processed.get('Kitchen area')

    if floor is not None and num_floors is not None:
        df_processed['floor_ratio'] = floor / num_floors.replace(0, 1)
        df_processed['is_first_floor'] = (floor == 1).astype(int)
        df_processed['is_last_floor'] = (floor == num_floors).astype(int)

    if living_area is not None and area is not None:
        df_processed['living_area_ratio'] = living_area / area.replace(0, 1)

    if kitchen_area is not None and area is not None:
        df_processed['kitchen_area_ratio'] = kitchen_area / area.replace(0, 1)
        
    # Удаление ненужных столбцов
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)

    # Заполнение пропусков
    cols_to_fill = ['Living area', 'Kitchen area', 'Floor', 'floor_ratio', 'living_area_ratio', 'kitchen_area_ratio']
    for col in cols_to_fill:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Заполнение пропусков в категориальных колонках модой
    for col in ['Apartment type', 'Renovation', 'Region', 'Metro station']:
         if col in df_processed.columns and df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # Применение сохраненных кодировщиков
    if 'one_hot_encoder' in st.session_state and 'ordinal_encoder' in st.session_state:
        ohe = st.session_state['one_hot_encoder']
        ordinal_encoder = st.session_state['ordinal_encoder']
        
        categorical_cols = ['Apartment type', 'Renovation']
        high_cardinality_cols = ['Metro station', 'Region']

        # One-Hot Encoding
        ohe_features = ohe.transform(df_processed[categorical_cols])
        ohe_df = pd.DataFrame(ohe_features, columns=ohe.get_feature_names_out(categorical_cols), index=df_processed.index)
        
        # Ordinal Encoding
        ordinal_features = ordinal_encoder.transform(df_processed[high_cardinality_cols])
        df_processed[high_cardinality_cols] = ordinal_features

        df_processed = pd.concat([df_processed.drop(columns=categorical_cols), ohe_df], axis=1)

    # Приведение колонок в соответствие с обучением
    if 'train_columns' in st.session_state:
        train_cols = st.session_state.get('train_columns', [])
        current_cols = df_processed.columns.tolist()

        missing_cols = set(train_cols) - set(current_cols)
        for c in missing_cols:
            df_processed[c] = 0

        extra_cols = set(current_cols) - set(train_cols)
        df_processed = df_processed.drop(columns=list(extra_cols), errors='ignore')

        df_processed = df_processed[train_cols]

    return df_processed

# ИСПРАВЛЕНИЕ: Функция для обучения моделей с настроенными гиперпараметрами
@st.cache_resource
def train_models(X_train, y_train):
    """Обучает модели регрессии с улучшенными параметрами."""
    models = {}

    with st.spinner('Обучение Random Forest...'):
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                 max_depth=15, min_samples_leaf=5, max_features=0.7)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf

    with st.spinner('Обучение XGBoost...'):
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300,
                                   learning_rate=0.05, max_depth=7, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, n_jobs=-1)
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
        buffer = io.StringIO()
        df_train_raw.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # --- РАЗДЕЛ 2: Обработка и обучение ---
    elif app_mode == "Обработка и обучение":
        st.header("2. Предобработка данных и обучение моделей")

        st.subheader("2.1. Очистка от выбросов")
        st.write("Проанализируем распределение цен и удалим выбросы.")

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

        # ИСПРАВЛЕНИЕ: Обучение и сохранение кодировщиков
        st.subheader("2.2. Обучение кодировщиков и предобработка")

        categorical_cols = ['Apartment type', 'Renovation']
        high_cardinality_cols = ['Metro station', 'Region']

        # Заполняем пропуски модой перед обучением кодировщиков
        df_train_filtered_imputed = df_train_filtered.copy()
        for col in categorical_cols + high_cardinality_cols:
             if df_train_filtered_imputed[col].isnull().any():
                df_train_filtered_imputed[col] = df_train_filtered_imputed[col].fillna(df_train_filtered_imputed[col].mode()[0])
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # -1 для неизвестных категорий

        ohe.fit(df_train_filtered_imputed[categorical_cols])
        ordinal_encoder.fit(df_train_filtered_imputed[high_cardinality_cols])
        
        st.session_state['one_hot_encoder'] = ohe
        st.session_state['ordinal_encoder'] = ordinal_encoder

        df_train_to_process = df_train_filtered.drop('Price', axis=1)

        # Временная запись колонок до их изменения кодировщиками
        # Мы сделаем это после создания признаков, но до one-hot
        temp_df_with_features = df_train_to_process.copy()
        floor = temp_df_with_features.get('Floor')
        num_floors = temp_df_with_features.get('Number of floors')
        if floor is not None and num_floors is not None:
             temp_df_with_features['floor_ratio'] = floor / num_floors.replace(0, 1)
             temp_df_with_features['is_first_floor'] = (floor == 1).astype(int)
             temp_df_with_features['is_last_floor'] = (floor == num_floors).astype(int)
        # Сохраняем названия колонок для будущей сверки
        ohe_feature_names = st.session_state['one_hot_encoder'].get_feature_names_out(categorical_cols).tolist()
        base_feature_names = [col for col in temp_df_with_features.columns if col not in categorical_cols]
        st.session_state['train_columns'] = base_feature_names + ohe_feature_names

        df_processed = preprocess_data_new(df_train_to_process)
        df_processed['Price'] = df_train_filtered['Price'].values

        st.write("Данные после обработки (создание признаков, кодирование):")
        st.dataframe(df_processed.head())

        X = df_processed.drop('Price', axis=1)
        y = df_processed['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("2.3. Обучение моделей и оценка")
        models = train_models(X_train, y_train)
        st.session_state['models'] = models
        st.session_state['data_prepared'] = True

        st.write("Оценка моделей на тестовой выборке (Mean Absolute Error):")
        col1, col2 = st.columns(2)
        
        maes = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            maes[name] = mae
            if name == 'Random Forest':
                col1.metric(label=name, value=f"{mae:,.0f} руб.")
            else:
                col2.metric(label=name, value=f"{mae:,.0f} руб.")
        
        best_model_name = min(maes, key=maes.get)
        best_model = models[best_model_name]
        st.success(f"Лучшая модель: **{best_model_name}**")

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

        if not df_valid_filtered.empty:
            y_valid = df_valid_filtered['Price']
            X_valid_raw = df_valid_filtered.drop('Price', axis=1)
            X_valid = preprocess_data_new(X_valid_raw)
            
            valid_pred = best_model.predict(X_valid)
            valid_mae = mean_absolute_error(y_valid, valid_pred)
            st.metric(label=f"MAE на валидационном наборе ({best_model_name})", value=f"{valid_mae:,.0f} руб.")
            st.info("Это финальная оценка качества модели на данных, которые она не видела во время обучения.")
        else:
            st.warning("В валидационном наборе не осталось данных после фильтрации по цене. Оценка невозможна.")
    
    # --- РАЗДЕЛ 3: Прогнозирование ---
    elif app_mode == "Прогнозирование цены":
        st.header("3. Сделать прогноз")

        if 'models' not in st.session_state or not st.session_state.get('data_prepared'):
            st.warning("Пожалуйста, сначала перейдите в раздел 'Обработка и обучение', чтобы подготовить данные и обучить модели.")
        else:
            best_model_name = st.session_state.get('best_model_name', "Random Forest")
            st.write(f"Введите параметры квартиры, чтобы получить предсказание от лучшей модели ({best_model_name}).")
            
            final_model = st.session_state['models'][best_model_name]
            st.sidebar.subheader("Параметры квартиры:")
            
            # Используем .dropna() и .unique() для получения чистых списков для селектбоксов
            apartment_types = sorted(df_train_raw['Apartment type'].dropna().unique())
            renovations = sorted(df_train_raw['Renovation'].dropna().unique())
            regions = sorted(df_train_raw['Region'].dropna().unique())
            metro_stations = sorted(df_train_raw['Metro station'].dropna().unique())

            with st.sidebar.form(key='prediction_form'):
                num_rooms = st.slider("Количество комнат", 1, 6, 2)
                area = st.slider("Общая площадь, м²", 20.0, 200.0, 55.0)
                living_area = st.slider("Жилая площадь, м²", 10.0, 150.0, 35.0)
                kitchen_area = st.slider("Площадь кухни, м²", 5.0, 50.0, 10.0)
                floor = st.slider("Этаж", 1, 50, 5)
                num_floors = st.slider("Всего этажей в доме", 1, 50, 12)
                minutes_to_metro = st.slider("Минут до метро", 1, 60, 15)
                
                apartment_type = st.selectbox("Тип апартаментов", apartment_types)
                renovation = st.selectbox("Ремонт", renovations)
                region = st.selectbox("Район", regions)
                metro_station = st.selectbox("Станция метро", metro_stations)
                
                submit_button = st.form_submit_button(label="Предсказать стоимость")

            if submit_button:
                input_data = {
                    'Minutes to metro': [minutes_to_metro], 'Number of rooms': [num_rooms],
                    'Area': [area], 'Living area': [living_area], 'Kitchen area': [kitchen_area],
                    'Floor': [floor], 'Number of floors': [num_floors],
                    'Metro station': [metro_station], 'Region': [region],
                    'Apartment type': [apartment_type], 'Renovation': [renovation]
                }
                input_df = pd.DataFrame(input_data)
                
                # ИСПРАВЛЕНИЕ: Используем новую, надежную функцию обработки
                processed_input = preprocess_data_new(input_df)

                if processed_input is not None:
                    prediction = final_model.predict(processed_input)[0]
                    st.success(f"## Предсказанная стоимость: **{prediction:,.0f} руб.**")
                else:
                    st.error("Ошибка при обработке введенных данных. Вернитесь в раздел 'Обработка и обучение'.")