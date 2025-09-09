import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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

#  УБРАН ДЕКОРАТОР @st.cache_data
def preprocess_data_new(_df, is_train=False):
    """Выполняет предобработку данных с использованием сохраненных кодировщиков и созданием признаков."""
    df_processed = _df.copy()

    # УЛУЧШЕНИЕ 1: Создание новых признаков (Feature Engineering)
    # Используем .get() для безопасного доступа к столбцам, которых может не быть в input_df
    floor = df_processed.get('Floor')
    num_floors = df_processed.get('Number of floors')
    living_area = df_processed.get('Living area')
    area = df_processed.get('Area')
    kitchen_area = df_processed.get('Kitchen area')

    if floor is not None and num_floors is not None:
        # Добавляем .replace(0, 1) чтобы избежать деления на ноль
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
    cols_to_fill = ['Living area', 'Kitchen area', 'Floor', 'Number of floors', 'floor_ratio', 'living_area_ratio', 'kitchen_area_ratio']
    for col in cols_to_fill:
        if col in df_processed.columns:
            # Для обучающей выборки считаем медиану, для остальных используем сохраненную
            if is_train:
                median_val = df_processed[col].median()
                if 'medians' not in st.session_state:
                    st.session_state['medians'] = {}
                st.session_state['medians'][col] = median_val
            else:
                median_val = st.session_state.get('medians', {}).get(col, df_processed[col].median())
            df_processed[col] = df_processed[col].fillna(median_val)

    # Заполнение пропусков в категориальных колонках модой
    for col in ['Apartment type', 'Renovation', 'Region', 'Metro station']:
        if col in df_processed.columns and df_processed[col].isnull().any():
            # Для обучающей выборки считаем моду, для остальных используем сохраненную
            if is_train:
                mode_val = df_processed[col].mode()[0]
                if 'modes' not in st.session_state:
                    st.session_state['modes'] = {}
                st.session_state['modes'][col] = mode_val
            else:
                mode_val = st.session_state.get('modes', {}).get(col, df_processed[col].mode()[0])
            df_processed[col] = df_processed[col].fillna(mode_val)


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

        # Удаляем целевую переменную, если она есть в списке колонок для обучения
        if 'Price' in train_cols:
            train_cols.remove('Price')

        missing_cols = set(train_cols) - set(current_cols)
        for c in missing_cols:
            df_processed[c] = 0

        extra_cols = set(current_cols) - set(train_cols)
        df_processed = df_processed.drop(columns=list(extra_cols), errors='ignore')

        df_processed = df_processed[train_cols]


    return df_processed

#  Функция для обучения моделей с настроенными гиперпараметрами
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

        #  Замена фильтрации на логарифмическую трансформацию
        st.subheader("2.1. Обработка целевой переменной (цены)")
        st.write("Цены на недвижимость имеют сильно смещенное распределение с 'длинным хвостом' очень дорогих квартир. Чтобы модель лучше справлялась с этим диапазоном, мы применим логарифмическую трансформацию к ценам.")

        col1, col2 = st.columns(2)
        fig, ax = plt.subplots()
        df_train_raw['Price'].hist(bins=100, ax=ax)
        ax.set_title("Распределение цен до трансформации")
        ax.set_xlabel("Цена")
        ax.set_ylabel("Частота")
        col1.pyplot(fig)

        # Применяем логарифмическую трансформацию
        y_log_transformed = np.log1p(df_train_raw['Price'])

        fig, ax = plt.subplots()
        y_log_transformed.hist(bins=100, ax=ax)
        ax.set_title("Распределение цен после логарифмирования")
        ax.set_xlabel("Log(1 + Цена)")
        ax.set_ylabel("Частота")
        col2.pyplot(fig)
        st.info("Как видно на графике, после трансформации распределение стало более 'нормальным', что помогает моделям машинного обучения работать эффективнее.")

        # Копируем датафрейм для дальнейшей обработки
        df_train_processed = df_train_raw.copy()
        df_train_processed['Price'] = y_log_transformed
        
        #  Обучение и сохранение кодировщиков
        st.subheader("2.2. Обучение кодировщиков и предобработка")

        categorical_cols = ['Apartment type', 'Renovation']
        high_cardinality_cols = ['Metro station', 'Region']

        # Для обучения кодировщиков используем копию данных, где пропуски заполнены модой
        df_train_for_encoding = df_train_raw.copy()
        for col in categorical_cols + high_cardinality_cols:
            if df_train_for_encoding[col].isnull().any():
                df_train_for_encoding[col] = df_train_for_encoding[col].fillna(df_train_for_encoding[col].mode()[0])

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # -1 для неизвестных категорий

        ohe.fit(df_train_for_encoding[categorical_cols])
        ordinal_encoder.fit(df_train_for_encoding[high_cardinality_cols])

        st.session_state['one_hot_encoder'] = ohe
        st.session_state['ordinal_encoder'] = ordinal_encoder

        # Предобработка тренировочных данных
        X_train_features = preprocess_data_new(df_train_processed.drop('Price', axis=1), is_train=True)
        y_train_log_values = df_train_processed['Price'].values

        # Сохраняем названия колонок для будущей сверки
        st.session_state['train_columns'] = X_train_features.columns.tolist()

        st.write("Данные для обучения после обработки:")
        st.dataframe(X_train_features.head())

        #  Разделяем на обучающую и тестовую выборки. y_train и y_test теперь содержат логарифм цены
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X_train_features, y_train_log_values, test_size=0.2, random_state=42
        )

        st.subheader("2.3. Обучение моделей и оценка")
        models = train_models(X_train, y_train_log)
        st.session_state['models'] = models
        st.session_state['data_prepared'] = True

        st.write("Оценка моделей на отложенной тестовой выборке (Mean Absolute Error):")
        col1, col2 = st.columns(2)

        maes = {}
        for name, model in models.items():
            # Предсказываем логарифм цены
            y_pred_log = model.predict(X_test)
            # Возвращаемся к исходному масштабу цен
            y_pred_original = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test_log)
            # Считаем ошибку в рублях
            mae = mean_absolute_error(y_test_original, y_pred_original)
            maes[name] = mae
            if name == 'Random Forest':
                col1.metric(label=name, value=f"{mae:,.0f} руб.")
            else:
                col2.metric(label=name, value=f"{mae:,.0f} руб.")

        best_model_name = min(maes, key=maes.get)
        best_model = models[best_model_name]
        st.session_state['best_model_name'] = best_model_name # Сохраняем имя лучшей модели
        st.success(f"Лучшая модель: **{best_model_name}**")

        st.subheader(f"2.4. Анализ лучшей модели: {best_model_name}")
        
        #  Для графика используем цены в исходном масштабе
        y_test_original = np.expm1(y_test_log)
        y_pred_original = np.expm1(best_model.predict(X_test))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_original, y_pred_original, alpha=0.3)
        ax.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], '--r', linewidth=2)
        ax.set_xlabel('Настоящая цена')
        ax.set_ylabel('Предсказанная цена')
        ax.set_title('Соответствие предсказаний и реальных цен')
        st.pyplot(fig)

        # Итоговая оценка на ВСЕЙ валидационной выборке
        st.subheader("2.5. Итоговая оценка на валидационных данных")
        st.info("Это финальная оценка качества модели на данных, которые она не видела во время обучения. Модель обучалась на всех данных с применением логарифмирования.")

        if not df_valid_raw.empty:
            y_valid_original = df_valid_raw['Price']
            X_valid_raw = df_valid_raw.drop('Price', axis=1)
            # Применяем предобработку к валидационным данным
            X_valid_processed = preprocess_data_new(X_valid_raw, is_train=False)

            valid_pred_log = best_model.predict(X_valid_processed)
            valid_pred_original = np.expm1(valid_pred_log)
            
            valid_mae = mean_absolute_error(y_valid_original, valid_pred_original)
            st.metric(label=f"MAE на валидационном наборе ({best_model_name})", value=f"{valid_mae:,.0f} руб.")

            # --- АНАЛИЗ ОШИБОК ---
            st.subheader("Анализ ошибок на валидационной выборке")
            st.write("Хотя модель теперь обучается на всех данных, ошибки для очень дорогих квартир все еще могут быть высокими. Посмотрим на 10 квартир с наибольшей ошибкой.")
            
            error_df = pd.DataFrame({
                'Реальная цена': y_valid_original,
                'Предсказанная цена': valid_pred_original,
                'Абсолютная ошибка': np.abs(y_valid_original - valid_pred_original)
            }).astype(int)

            st.dataframe(error_df.sort_values(by='Абсолютная ошибка', ascending=False).head(10))
        else:
            st.warning("Валидационный набор данных пуст. Оценка невозможна.")

    # --- РАЗДЕЛ 3: Прогнозирование ---
    elif app_mode == "Прогнозирование цены":
        st.header("3. Сделать прогноз")

        if 'models' not in st.session_state or not st.session_state.get('data_prepared'):
            st.warning("Пожалуйста, сначала перейдите в раздел 'Обработка и обучение', чтобы подготовить данные и обучить модели.")
        else:
            # Используем сохраненное имя лучшей модели
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


                processed_input = preprocess_data_new(input_df, is_train=False)

                if processed_input is not None:
                    
                    prediction_log = final_model.predict(processed_input)[0]
                    prediction_original = np.expm1(prediction_log)
                    
                    st.success(f"## Предсказанная стоимость: **{prediction_original:,.0f} руб.**")
                else:
                    st.error("Ошибка при обработке введенных данных. Вернитесь в раздел 'Обработка и обучение'.")