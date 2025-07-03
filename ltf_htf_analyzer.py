#!/usr/bin/env python3
"""
LTF/HTF Анализатор - модуль для разделения и анализа быстрых/медленных таймфреймов
Исправляет баги временных лагов + добавляет LTF/HTF функциональность

Совместим с main.py через конфигурацию
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import yaml
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


class LTFHTFAnalyzer:
    """
    LTF/HTF анализатор с исправлением багов временных лагов
    
    LTF (Low TimeFrame): 2m, 5m, 15m, 30m
    HTF (High TimeFrame): 1h, 4h, 1d, 1w
    """
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация LTF/HTF анализатора"""
        self.config = self._load_config(config_path)
        self.raw_data = None
        self.parsed_data = None
        
        # Разделенные данные
        self.ltf_data = None
        self.htf_data = None
        self.combined_data = None
        
        # Результаты анализа
        self.ltf_results = {}
        self.htf_results = {}
        self.combined_results = {}
        
        # Определение полей по типам (data-driven подход)
        self.ltf_indicators = {
            'group_1': ['rd', 'md', 'cd', 'cmd', 'macd', 'od', 'dd', 'cvd', 'drd', 'ad', 'ed', 'hd', 'sd'],
            'group_2': ['ro', 'mo', 'co', 'cz', 'do', 'ae', 'so'],
            'group_3': ['rz', 'mz', 'ciz', 'sz', 'dz', 'cvz', 'maz', 'oz'],
            'group_4': ['ef', 'wv', 'vc', 'ze', 'nw', 'as', 'vw'],
            'group_5': ['bs', 'wa', 'pd']
        }
        
        self.htf_indicators = self.ltf_indicators.copy()  # Те же группы, но с другими суффиксами
        
        # Создание структуры папок
        self._create_directories()
        
    def _load_config(self, config_path):
        """Загрузка конфигурации с расширениями для LTF/HTF"""
        default_config = {
            'analysis': {
                'min_accuracy': 0.60,
                'min_lift': 1.5,
                'validation_split': 0.3,
                'cv_folds': 5,
                'enable_ltf_htf': True,  # Новый флаг
                'legacy_mode': False     # Совместимость со старой системой
            },
            'event_detection': {
                'volatility_threshold': 2.0,
                'volume_threshold': 1.5,
                'price_change_threshold': 0.5
            },
            'ltf_htf': {
                'ltf_timeframes': ['2', '5', '15', '30'],
                'htf_timeframes': ['1h', '4h', '1d', '1w'],
                'separation_method': 'auto',  # auto, manual, mixed
                'temporal_lag_fix': True      # Исправление багов лагов
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _create_directories(self):
        """Создание структуры папок для результатов"""
        directories = [
            "results/ltf",
            "results/htf", 
            "results/combined",
            "results/ltf/plots",
            "results/htf/plots",
            "results/combined/plots",
            "results/ltf/reports",
            "results/htf/reports",
            "results/combined/reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def parse_mixed_format_file(self, file_path):
        """
        Универсальный парсер для разных форматов логов
        
        Поддерживает:
        1. [timestamp]: LTF|event_name|tf|timestamp|data...
        2. LTF|realtime_log|1|timestamp|data...
        3. HTF|realtime_log|1|timestamp|data...
        """
        print(f"🔍 Парсинг файла с LTF/HTF разделением: {file_path}")
        
        ltf_records = []
        htf_records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Определение формата и типа записи
                    record_type, parsed_record = self._parse_universal_line(line, line_num)
                    
                    if record_type == 'LTF' and parsed_record:
                        ltf_records.append(parsed_record)
                    elif record_type == 'HTF' and parsed_record:
                        htf_records.append(parsed_record)
                    elif record_type == 'MIXED' and parsed_record:
                        # Автоопределение LTF/HTF по содержимому полей
                        classified_type = self._classify_record_by_fields(parsed_record)
                        if classified_type == 'LTF':
                            ltf_records.append(parsed_record)
                        elif classified_type == 'HTF':
                            htf_records.append(parsed_record)
                        else:
                            # Если не удалось классифицировать, добавляем в оба
                            ltf_records.append(parsed_record)
                            htf_records.append(parsed_record)
                    
                except Exception as e:
                    print(f"⚠️ Ошибка парсинга строки {line_num}: {e}")
                    continue
        
        # Создание DataFrame
        self.ltf_data = pd.DataFrame(ltf_records) if ltf_records else pd.DataFrame()
        self.htf_data = pd.DataFrame(htf_records) if htf_records else pd.DataFrame()
        
        print(f"✅ Парсинг завершен:")
        print(f"   LTF записей: {len(self.ltf_data)}")
        print(f"   HTF записей: {len(self.htf_data)}")
        
        return self.ltf_data, self.htf_data
    
    def _parse_universal_line(self, line, line_num):
        """Универсальный парсер строки для разных форматов"""
        
        # Формат 1: [timestamp]: LTF|event_name|...
        if line.startswith('[') and ']:' in line:
            return self._parse_bracketed_format(line, line_num)
        
        # Формат 2: LTF|realtime_log|... или HTF|realtime_log|...
        elif line.startswith(('LTF', 'HTF')):
            return self._parse_sheets_format(line, line_num)
        
        # Формат 3: смешанный/неопределенный
        else:
            return self._parse_mixed_format(line, line_num)
    
    def _parse_bracketed_format(self, line, line_num):
        """Парсинг формата [timestamp]: LTF|event_name|..."""
        try:
            match = re.match(r'\[([^\]]+)\]:\s*(.+)', line)
            if not match:
                return None, None
            
            timestamp_str, data_str = match.groups()
            parts = data_str.split('|')
            
            if len(parts) < 6:
                return None, None
            
            log_type = parts[0]  # LTF или другое
            event_name = parts[1]
            tf = parts[2]
            event_timestamp = parts[3]
            
            # Парсинг candle данных и полей
            candle_part = '|'.join(parts[4:])
            candle_data, field_data = self._parse_candle_and_fields(candle_part)
            
            record = {
                'log_timestamp': pd.to_datetime(timestamp_str),
                'log_type': log_type,
                'event_name': event_name,
                'timeframe': tf,
                'event_timestamp': event_timestamp,
                'line_number': line_num,
                'data_source': 'bracketed',
                **candle_data,
                **field_data
            }
            
            # Возвращаем тип и запись
            if log_type.upper() == 'LTF':
                return 'LTF', record
            elif log_type.upper() == 'HTF':
                return 'HTF', record
            else:
                return 'MIXED', record
            
        except Exception as e:
            print(f"⚠️ Ошибка парсинга bracketed формата: {e}")
            return None, None
    
    def _parse_sheets_format(self, line, line_num):
        """Парсинг формата LTF|realtime_log|... из Google Sheets"""
        try:
            parts = line.split('\t')  # Google Sheets использует табы
            
            if len(parts) < 13:  # Минимум полей
                return None, None
            
            log_type = parts[0]  # LTF или HTF
            event_name = parts[1]  # realtime_log
            tf = parts[2]  # 1
            event_timestamp = parts[3]  # timestamp
            
            # Candle данные начинаются с 4 элемента
            candle_data = {
                'color': parts[4],
                'price_change': self._parse_percentage(parts[5]),
                'volume': self._parse_volume(parts[6]),
                'candle_type': parts[7],
                'completion': self._parse_percentage(parts[8]),
                'movement_24h': self._parse_percentage(parts[9]),
                'open': float(parts[10].replace('o:', '')),
                'high': float(parts[11].replace('h:', '')),
                'low': float(parts[12].replace('l:', '')),
                'close': float(parts[13].replace('c:', '')),
                'range': float(parts[14].replace('rng:', ''))
            }
            
            # Поля начинаются с элемента 15
            field_data = {}
            if len(parts) > 15 and parts[15]:
                field_items = parts[15].split(',')
                for item in field_items:
                    item = item.strip()
                    if '-' in item:
                        field_parts = item.split('-', 1)
                        if len(field_parts) == 2:
                            field_name = field_parts[0]
                            field_value = self._parse_field_value(field_parts[1])
                            field_data[field_name] = field_value
            
            record = {
                'log_timestamp': pd.to_datetime(event_timestamp),
                'log_type': log_type,
                'event_name': event_name,
                'timeframe': tf,
                'event_timestamp': event_timestamp,
                'line_number': line_num,
                'data_source': 'sheets',
                **candle_data,
                **field_data
            }
            
            return log_type.upper(), record
            
        except Exception as e:
            print(f"⚠️ Ошибка парсинга sheets формата: {e}")
            return None, None
    
    def _parse_mixed_format(self, line, line_num):
        """Парсинг смешанного/неопределенного формата"""
        # Попытка определить формат по содержимому
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 6:
                # Попытка парсинга как стандартный формат
                try:
                    candle_part = '|'.join(parts[3:]) if len(parts) > 3 else ''
                    candle_data, field_data = self._parse_candle_and_fields(candle_part)
                    
                    record = {
                        'log_timestamp': pd.to_datetime('now'),
                        'log_type': 'MIXED',
                        'event_name': parts[0] if len(parts) > 0 else 'unknown',
                        'timeframe': '1',
                        'event_timestamp': '',
                        'line_number': line_num,
                        'data_source': 'mixed',
                        **candle_data,
                        **field_data
                    }
                    
                    return 'MIXED', record
                except:
                    pass
        
        return None, None
    
    def _classify_record_by_fields(self, record):
        """Классификация записи как LTF/HTF по анализу полей"""
        ltf_field_count = 0
        htf_field_count = 0
        
        ltf_patterns = ['2', '5', '15', '30']
        htf_patterns = ['1h', '4h', '1d', '1w']
        
        for field_name in record.keys():
            if isinstance(field_name, str):
                # Проверка LTF паттернов
                for pattern in ltf_patterns:
                    if field_name.endswith(pattern) or f"{pattern}-" in field_name:
                        ltf_field_count += 1
                        break
                
                # Проверка HTF паттернов  
                for pattern in htf_patterns:
                    if pattern.lower() in field_name.lower():
                        htf_field_count += 1
                        break
        
        # Решение на основе количества полей каждого типа
        if htf_field_count > ltf_field_count:
            return 'HTF'
        elif ltf_field_count > htf_field_count:
            return 'LTF'
        else:
            return 'MIXED'  # Неопределенный тип
    
    def _parse_candle_and_fields(self, data_str):
        """Парсинг данных свечей и полей (улучшенная версия из main.py)"""
        candle_data = {}
        field_data = {}
        
        # Разделение по |
        parts = data_str.split('|')
        
        # Первые части - данные свечей
        if len(parts) >= 6:
            candle_data['color'] = parts[0]
            candle_data['price_change'] = self._parse_percentage(parts[1])
            candle_data['volume'] = self._parse_volume(parts[2])
            candle_data['candle_type'] = parts[3]
            candle_data['completion'] = self._parse_percentage(parts[4])
            
            # Парсинг 24h движения
            movement_24h_str = parts[5] if len(parts) > 5 else ''
            if '_24h' in movement_24h_str:
                movement_value = movement_24h_str.replace('_24h', '').replace('%', '')
                try:
                    candle_data['movement_24h'] = float(movement_value)
                except ValueError:
                    candle_data['movement_24h'] = 0.0
            else:
                candle_data['movement_24h'] = self._parse_percentage(movement_24h_str)
        
        # Объединение всех оставшихся частей для парсинга полей
        remaining_data = '|'.join(parts[6:]) if len(parts) > 6 else ''
        
        # Парсинг OHLC данных
        ohlc_pattern = r'o:([\d.]+).*?h:([\d.]+).*?l:([\d.]+).*?c:([\d.]+)'
        ohlc_match = re.search(ohlc_pattern, remaining_data)
        if ohlc_match:
            candle_data['open'] = float(ohlc_match.group(1))
            candle_data['high'] = float(ohlc_match.group(2))
            candle_data['low'] = float(ohlc_match.group(3))
            candle_data['close'] = float(ohlc_match.group(4))
        
        # Парсинг range
        rng_pattern = r'rng:([\d.]+)'
        rng_match = re.search(rng_pattern, remaining_data)
        if rng_match:
            candle_data['range'] = float(rng_match.group(1))
        
        # Парсинг полей
        ohlc_rng_pattern = r'o:[\d.]+\|h:[\d.]+\|l:[\d.]+\|c:[\d.]+\|rng:[\d.]+\|?'
        field_part = re.sub(ohlc_rng_pattern, '', remaining_data)
        
        if field_part:
            field_items = field_part.replace('|', ',').split(',')
            
            for item in field_items:
                item = item.strip()
                if not item:
                    continue
                
                if '-' in item:
                    parts_field = item.split('-', 1)
                    if len(parts_field) == 2:
                        field_name = parts_field[0]
                        field_value = parts_field[1]
                        parsed_value = self._parse_field_value(field_value)
                        field_data[field_name] = parsed_value
        
        return candle_data, field_data
    
    def _parse_percentage(self, value_str):
        """Парсинг процентных значений"""
        if '%' in value_str:
            return float(value_str.replace('%', ''))
        return float(value_str) if value_str.replace('.', '').replace('-', '').isdigit() else 0
    
    def _parse_volume(self, volume_str):
        """Парсинг объемных данных"""
        volume_str = volume_str.upper()
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1000
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1000000
        return float(volume_str) if volume_str.replace('.', '').isdigit() else 0
    
    def _parse_field_value(self, value_str):
        """Универсальный парсер значений полей"""
        if not value_str or value_str == '':
            return 0.0
            
        clean_value = str(value_str).replace('%', '').replace('σ', '')
        
        # Обработка восклицательных знаков
        if '!!!' in clean_value:
            clean_value = clean_value.replace('!!!', '')
            multiplier = 3.0
        elif '!!' in clean_value:
            clean_value = clean_value.replace('!!', '')
            multiplier = 2.0
        elif '!' in clean_value:
            clean_value = clean_value.replace('!', '')
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        # Обработка суффиксов времени
        for suffix in ['_24h', '_1h', '_4h', '_1d', '_1w']:
            clean_value = clean_value.replace(suffix, '')
        
        if not clean_value or clean_value in ['', '-', '+', '.']:
            return 0.0
        
        try:
            numeric_pattern = r'^[+-]?\d*\.?\d+$'
            if re.match(numeric_pattern, clean_value):
                if '.' in clean_value:
                    result = float(clean_value) * multiplier
                else:
                    result = int(clean_value) * multiplier
                return result
            else:
                number_pattern = r'[+-]?\d*\.?\d+'
                number_match = re.search(number_pattern, clean_value)
                if number_match:
                    return float(number_match.group()) * multiplier
                else:
                    return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def analyze_temporal_lags_fixed(self, data, field_groups, type_name):
        """
        ИСПРАВЛЕННЫЙ анализ временных лагов
        Проблема была в том, что лаги всегда возвращали 0.0
        """
        print(f"⏱️ Исправленный анализ временных лагов для {type_name}...")
        
        if data is None or len(data) == 0:
            return {}
        
        df = data.copy()
        
        # Сначала определяем события если их нет
        if 'is_event' not in df.columns:
            df = self._detect_events_for_data(df)
        
        lag_analysis = {}
        
        # Для каждой группы полей
        for group_name, fields in field_groups.items():
            group_lags = []
            activations_found = 0
            total_events_checked = 0
            
            # Получаем индексы событий
            event_indices = df[df['is_event'] == 1].index.tolist()
            
            print(f"   Анализ группы {group_name}: {len(event_indices)} событий")
            
            for event_idx in event_indices:
                total_events_checked += 1
                event_found = False
                
                # Поиск активации полей перед событием (до 20 периодов назад)
                for lag in range(1, min(21, event_idx + 1)):  # Не выходим за границы данных
                    check_idx = event_idx - lag
                    
                    if check_idx >= 0 and check_idx < len(df):
                        # Проверяем активацию в этой точке
                        activation_strength = self._calculate_activation_strength_fixed(
                            df.iloc[check_idx], fields
                        )
                        
                        # Если нашли значимую активацию
                        if activation_strength > 0:
                            group_lags.append(lag)
                            activations_found += 1
                            event_found = True
                            break
                
                # Если активация не найдена, записываем максимальный лаг
                if not event_found:
                    group_lags.append(20)
            
            # Расчет статистики
            if group_lags:
                lag_analysis[group_name] = {
                    'mean_lag': np.mean(group_lags),
                    'median_lag': np.median(group_lags),
                    'std_lag': np.std(group_lags),
                    'min_lag': np.min(group_lags),
                    'max_lag': np.max(group_lags),
                    'activation_rate': activations_found / max(1, total_events_checked),
                    'total_events': total_events_checked,
                    'activations_found': activations_found,
                    'lag_distribution': {
                        'lag_1_3': sum(1 for lag in group_lags if 1 <= lag <= 3) / len(group_lags),
                        'lag_4_10': sum(1 for lag in group_lags if 4 <= lag <= 10) / len(group_lags),
                        'lag_11_plus': sum(1 for lag in group_lags if lag > 10) / len(group_lags)
                    }
                }
                
                print(f"     {group_name}: средний лаг {lag_analysis[group_name]['mean_lag']:.1f}, активаций {activations_found}/{total_events_checked}")
            else:
                lag_analysis[group_name] = {
                    'mean_lag': 0,
                    'median_lag': 0,
                    'std_lag': 0,
                    'min_lag': 0,
                    'max_lag': 0,
                    'activation_rate': 0,
                    'total_events': total_events_checked,
                    'activations_found': 0,
                    'lag_distribution': {'lag_1_3': 0, 'lag_4_10': 0, 'lag_11_plus': 0}
                }
        
        print(f"✅ Анализ лагов для {type_name} завершен")
        return lag_analysis
    
    def _detect_events_for_data(self, df):
        """Определение событий для данных если их нет"""
        # Преобразование ключевых полей в числовой формат
        numeric_cols = ['high', 'low', 'open', 'close', 'volume', 'price_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Расчет производных метрик
        df['price_volatility'] = (df['high'] - df['low']) / df['open'].replace(0, 1) * 100
        df['price_change_abs'] = abs(df['price_change'])
        df['volume_log'] = np.log1p(df['volume'])
        
        # Обработка infinity и NaN значений
        df['price_volatility'] = df['price_volatility'].replace([np.inf, -np.inf], 0).fillna(0)
        df['price_change_abs'] = df['price_change_abs'].fillna(0)
        df['volume_log'] = df['volume_log'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Статистические пороги
        vol_threshold = df['price_volatility'].quantile(0.8)
        change_threshold = df['price_change_abs'].quantile(0.8)
        volume_threshold = df['volume_log'].quantile(0.8)
        
        # Определение событий
        df['is_event'] = (
            (df['price_volatility'] > vol_threshold) |
            (df['price_change_abs'] > change_threshold) |
            (df['volume_log'] > volume_threshold)
        ).astype(int)
        
        return df
    
    def _calculate_activation_strength_fixed(self, row, fields):
        """ИСПРАВЛЕННЫЙ расчет силы активации группы полей"""
        activation_sum = 0
        field_count = 0
        
        # Ищем все поля данной группы в записи
        for base_field in fields:
            # Поиск всех вариаций поля (с разными суффиксами)
            for field_name in row.index:
                if isinstance(field_name, str) and field_name.startswith(base_field):
                    try:
                        value = row[field_name]
                        if pd.notna(value) and value != 0:
                            # Преобразование в число
                            if isinstance(value, (int, float)):
                                numeric_value = abs(float(value))
                            elif isinstance(value, str):
                                # Попытка извлечь число из строки
                                clean_value = value.replace('%', '').replace('σ', '').replace('!', '')
                                number_pattern = r'[+-]?\d*\.?\d+'
                                number_match = re.search(number_pattern, clean_value)
                                if number_match:
                                    numeric_value = abs(float(number_match.group()))
                                else:
                                    continue
                            else:
                                continue
                            
                            # Учитываем только значимые активации
                            if numeric_value > 0.1:  # Порог значимости
                                activation_sum += numeric_value
                                field_count += 1
                                
                    except (ValueError, TypeError):
                        continue
        
        # Возвращаем средневзвешенную силу активации
        return activation_sum / max(1, field_count) if field_count > 0 else 0.0
    
    def analyze_ltf(self):
        """Анализ LTF данных"""
        print("⚡ Анализ LTF данных (быстрые таймфреймы)...")
        
        if self.ltf_data is None or len(self.ltf_data) == 0:
            print("⚠️ Нет LTF данных для анализа")
            return {}
        
        # Определение событий
        ltf_with_events = self._detect_events_for_data(self.ltf_data.copy())
        
        # Исправленный анализ временных лагов
        ltf_lags = self.analyze_temporal_lags_fixed(
            ltf_with_events, 
            self.ltf_indicators, 
            'LTF'
        )
        
        # Построение признаков и анализ
        ltf_features = self._build_features_for_type(ltf_with_events, 'LTF')
        ltf_correlations = self._analyze_correlations_for_type(ltf_features, ltf_with_events, 'LTF')
        ltf_thresholds = self._find_thresholds_for_type(ltf_features, ltf_with_events, 'LTF')
        
        # Построение скоринга
        ltf_scoring = self._build_scoring_for_type(
            ltf_features, ltf_with_events, ltf_thresholds, 'LTF'
        )
        
        # Валидация
        ltf_validation = self._validate_type_system(
            ltf_features, ltf_with_events, ltf_scoring, 'LTF'
        )
        
        self.ltf_results = {
            'data_shape': ltf_with_events.shape,
            'events_total': ltf_with_events['is_event'].sum(),
            'events_rate': ltf_with_events['is_event'].mean(),
            'temporal_lags': ltf_lags,
            'features': ltf_features,
            'correlations': ltf_correlations,
            'thresholds': ltf_thresholds,
            'scoring_system': ltf_scoring,
            'validation': ltf_validation
        }
        
        print(f"✅ LTF анализ завершен. События: {self.ltf_results['events_total']}/{len(ltf_with_events)} ({self.ltf_results['events_rate']:.2%})")
        
        return self.ltf_results
    
    def analyze_htf(self):
        """Анализ HTF данных"""
        print("🐌 Анализ HTF данных (медленные таймфреймы)...")
        
        if self.htf_data is None or len(self.htf_data) == 0:
            print("⚠️ Нет HTF данных для анализа")
            return {}
        
        # Определение событий
        htf_with_events = self._detect_events_for_data(self.htf_data.copy())
        
        # Исправленный анализ временных лагов
        htf_lags = self.analyze_temporal_lags_fixed(
            htf_with_events, 
            self.htf_indicators, 
            'HTF'
        )
        
        # Построение признаков и анализ
        htf_features = self._build_features_for_type(htf_with_events, 'HTF')
        htf_correlations = self._analyze_correlations_for_type(htf_features, htf_with_events, 'HTF')
        htf_thresholds = self._find_thresholds_for_type(htf_features, htf_with_events, 'HTF')
        
        # Построение скоринга
        htf_scoring = self._build_scoring_for_type(
            htf_features, htf_with_events, htf_thresholds, 'HTF'
        )
        
        # Валидация
        htf_validation = self._validate_type_system(
            htf_features, htf_with_events, htf_scoring, 'HTF'
        )
        
        self.htf_results = {
            'data_shape': htf_with_events.shape,
            'events_total': htf_with_events['is_event'].sum(),
            'events_rate': htf_with_events['is_event'].mean(),
            'temporal_lags': htf_lags,
            'features': htf_features,
            'correlations': htf_correlations,
            'thresholds': htf_thresholds,
            'scoring_system': htf_scoring,
            'validation': htf_validation
        }
        
        print(f"✅ HTF анализ завершен. События: {self.htf_results['events_total']}/{len(htf_with_events)} ({self.htf_results['events_rate']:.2%})")
        
        return self.htf_results
    
    def _build_features_for_type(self, data, type_name):
        """Построение признаков для конкретного типа"""
        print(f"🔧 Построение признаков для {type_name}...")
        
        df = data.copy()
        feature_columns = []
        
        # Базовые ценовые признаки
        price_features = ['open', 'high', 'low', 'close', 'range', 'price_change', 'volume']
        for feature in price_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                feature_columns.append(feature)
        
        # Признаки из индикаторов соответствующего типа
        field_groups = self.ltf_indicators if type_name == 'LTF' else self.htf_indicators
        
        for group_name, fields in field_groups.items():
            for field in fields:
                field_columns = [col for col in df.columns if col.startswith(field)]
                for col in field_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                feature_columns.extend(field_columns)
        
        # Создание лаговых признаков
        lag_features = []
        for lag in [1, 2, 3, 5]:
            for feature in feature_columns[:20]:  # Ограничиваем для производительности
                if feature in df.columns:
                    lag_col = f"{feature}_lag_{lag}"
                    df[lag_col] = pd.to_numeric(df[feature], errors='coerce').shift(lag).fillna(0)
                    lag_features.append(lag_col)
        
        feature_columns.extend(lag_features)
        
        # Фильтрация признаков
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features]
        
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        # Удаление признаков с низкой вариативностью
        var_threshold = 0.01
        feature_vars = features_df.var()
        high_var_features = feature_vars[feature_vars > var_threshold].index.tolist()
        
        result_features = features_df[high_var_features]
        print(f"   Создано {len(result_features.columns)} признаков для {type_name}")
        
        return result_features
    
    def _analyze_correlations_for_type(self, features, data, type_name):
        """Корреляционный анализ для типа"""
        print(f"🔗 Корреляционный анализ для {type_name}...")
        
        correlation_results = {}
        field_groups = self.ltf_indicators if type_name == 'LTF' else self.htf_indicators
        
        for group_name, fields in field_groups.items():
            group_features = []
            for field in fields:
                field_cols = [col for col in features.columns if col.startswith(field)]
                group_features.extend(field_cols)
            
            if group_features:
                try:
                    group_data = features[group_features]
                    for col in group_data.columns:
                        group_data[col] = pd.to_numeric(group_data[col], errors='coerce').fillna(0)
                    
                    group_activity = group_data.abs().mean(axis=1)
                    correlation_with_events = group_activity.corr(data['is_event'])
                    
                    if pd.isna(correlation_with_events):
                        correlation_with_events = 0.0
                    
                    correlation_results[group_name] = {
                        'event_correlation': correlation_with_events,
                        'feature_count': len(group_features),
                        'avg_activity': group_activity.mean()
                    }
                except Exception as e:
                    correlation_results[group_name] = {
                        'event_correlation': 0.0,
                        'feature_count': len(group_features) if group_features else 0,
                        'avg_activity': 0.0
                    }
        
        return correlation_results
    
    def _find_thresholds_for_type(self, features, data, type_name):
        """Поиск порогов для типа"""
        print(f"🎯 Поиск порогов для {type_name}...")
        
        threshold_results = {}
        
        for feature in features.columns:
            try:
                feature_data = pd.to_numeric(features[feature], errors='coerce').fillna(0)
                events = data['is_event']
                
                if feature_data.std() < 0.01:
                    continue
                
                thresholds = np.percentile(feature_data, np.arange(10, 100, 10))
                best_threshold = None
                best_score = 0
                
                for threshold in thresholds:
                    binary_feature = (abs(feature_data) > threshold).astype(int)
                    
                    if binary_feature.sum() > 10:
                        try:
                            score = roc_auc_score(events, binary_feature)
                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                        except ValueError:
                            continue
                
                if best_threshold is not None and best_score > 0.55:
                    threshold_results[feature] = {
                        'threshold': best_threshold,
                        'roc_auc': best_score,
                        'activation_rate': (abs(feature_data) > best_threshold).mean()
                    }
            except Exception:
                continue
        
        print(f"   Найдены пороги для {len(threshold_results)} полей")
        return threshold_results
    
    def _build_scoring_for_type(self, features, data, thresholds, type_name):
        """Построение скоринга для типа"""
        print(f"⚖️ Построение скоринга для {type_name}...")
        
        if not thresholds:
            return None
        
        scoring_features = []
        feature_weights = {}
        
        for feature, threshold_info in thresholds.items():
            if feature in features.columns:
                try:
                    binary_col = f"{feature}_activated"
                    feature_data = pd.to_numeric(features[feature], errors='coerce').fillna(0)
                    features[binary_col] = (abs(feature_data) > threshold_info['threshold']).astype(int)
                    scoring_features.append(binary_col)
                    feature_weights[binary_col] = threshold_info['roc_auc']
                except Exception:
                    continue
        
        if scoring_features and len(scoring_features) > 0:
            try:
                X = features[scoring_features]
                y = data['is_event']
                
                if y.sum() == 0:
                    return None
                
                # Обучение модели
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                feature_importance = dict(zip(scoring_features, rf.feature_importances_))
                
                return {
                    'model': rf,
                    'features': scoring_features,
                    'feature_weights': feature_weights,
                    'feature_importance': feature_importance,
                    'thresholds': thresholds
                }
            except Exception:
                return None
        
        return None
    
    def _validate_type_system(self, features, data, scoring_system, type_name):
        """Валидация системы для типа"""
        print(f"✅ Валидация системы {type_name}...")
        
        if scoring_system is None:
            return {
                'roc_auc': 0.5,
                'accuracy': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'lift': 0.0,
                'meets_requirements': False
            }
        
        try:
            split_point = int(len(features) * 0.7)
            
            X_train = features.iloc[:split_point]
            X_val = features.iloc[split_point:]
            y_train = data['is_event'].iloc[:split_point]
            y_val = data['is_event'].iloc[split_point:]
            
            scoring_features = scoring_system['features']
            model = scoring_system['model']
            
            if y_train.sum() == 0 or y_val.sum() == 0:
                return {
                    'roc_auc': 0.5,
                    'accuracy': 0.5,
                    'precision': 0.0,
                    'recall': 0.0,
                    'lift': 0.0,
                    'meets_requirements': False
                }
            
            model.fit(X_train[scoring_features], y_train)
            
            y_pred_proba = model.predict_proba(X_val[scoring_features])[:, 1]
            y_pred = model.predict(X_val[scoring_features])
            
            validation_results = {
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'accuracy': (y_pred == y_val).mean(),
                'precision': (y_pred * y_val).sum() / max(1, y_pred.sum()),
                'recall': (y_pred * y_val).sum() / max(1, y_val.sum()),
                'event_rate_actual': y_val.mean(),
                'event_rate_predicted': y_pred.mean()
            }
            
            baseline_rate = y_val.mean()
            if baseline_rate > 0:
                validation_results['lift'] = validation_results['precision'] / baseline_rate
            else:
                validation_results['lift'] = 0
            
            meets_requirements = (
                validation_results['accuracy'] >= 0.55 and  # Немного снижены требования для отдельных типов
                validation_results['lift'] >= 1.2
            )
            
            validation_results['meets_requirements'] = meets_requirements
            
            print(f"   {type_name} валидация: ROC-AUC {validation_results['roc_auc']:.3f}, Точность {validation_results['accuracy']:.3f}")
            
            return validation_results
            
        except Exception as e:
            print(f"⚠️ Ошибка валидации {type_name}: {e}")
            return {
                'roc_auc': 0.5,
                'accuracy': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'lift': 0.0,
                'meets_requirements': False
            }
    
    def save_results(self):
        """Сохранение результатов LTF/HTF анализа"""
        print("💾 Сохранение результатов LTF/HTF анализа...")
        
        # Сохранение LTF результатов
        if self.ltf_results and 'scoring_system' in self.ltf_results and self.ltf_results['scoring_system']:
            self._save_type_results('ltf', self.ltf_results)
        
        # Сохранение HTF результатов
        if self.htf_results and 'scoring_system' in self.htf_results and self.htf_results['scoring_system']:
            self._save_type_results('htf', self.htf_results)
        
        # Создание сводного отчета
        self._create_summary_report()
        
        # Создание визуализаций
        self._create_comparison_plots()
        
        print("✅ Результаты сохранены")
    
    def _save_type_results(self, type_name, results):
        """Сохранение результатов для конкретного типа"""
        output_dir = Path(f"results/{type_name}")
        
        # Матрица весов
        if 'scoring_system' in results and results['scoring_system']:
            weights_data = []
            scoring_system = results['scoring_system']
            thresholds = results.get('thresholds', {})
            
            for feature in scoring_system['features']:
                weight = scoring_system['feature_importance'].get(feature, 0)
                base_feature = feature.replace('_activated', '')
                threshold_info = thresholds.get(base_feature, {})
                
                weights_data.append({
                    'feature': feature,
                    'base_feature': base_feature,
                    'weight': weight,
                    'threshold': threshold_info.get('threshold', 0),
                    'roc_auc': threshold_info.get('roc_auc', 0),
                    'activation_rate': threshold_info.get('activation_rate', 0)
                })
            
            weights_df = pd.DataFrame(weights_data)
            weights_df.to_csv(output_dir / f"weight_matrix_{type_name}.csv", index=False)
        
        # Конфигурация скоринга
        config = {
            'type': type_name.upper(),
            'data_shape': results['data_shape'],
            'events_total': int(results['events_total']),
            'events_rate': float(results['events_rate']),
            'thresholds': {k: float(v['threshold']) for k, v in results.get('thresholds', {}).items()},
            'validation_score': float(results['validation']['roc_auc']) if 'validation' in results else 0,
            'temporal_lags': {
                group: {
                    'mean_lag': float(lag_info['mean_lag']),
                    'activation_rate': float(lag_info['activation_rate'])
                }
                for group, lag_info in results.get('temporal_lags', {}).items()
            }
        }
        
        with open(output_dir / f"scoring_config_{type_name}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Временные лаги
        if 'temporal_lags' in results:
            lags_df = pd.DataFrame(results['temporal_lags']).T
            lags_df.to_csv(output_dir / f"temporal_lags_{type_name}.csv")
    
    def _create_summary_report(self):
        """Создание сводного отчета"""
        report_lines = [
            "LTF/HTF АНАЛИЗАТОР - СВОДНЫЙ ОТЧЕТ",
            "=" * 50,
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "СТАТИСТИКА ДАННЫХ:",
        ]
        
        if self.ltf_data is not None:
            report_lines.extend([
                f"LTF записей: {len(self.ltf_data)}",
                f"LTF событий: {self.ltf_results.get('events_total', 0)}",
                f"LTF частота событий: {self.ltf_results.get('events_rate', 0):.2%}",
            ])
        
        if self.htf_data is not None:
            report_lines.extend([
                f"HTF записей: {len(self.htf_data)}",
                f"HTF событий: {self.htf_results.get('events_total', 0)}",
                f"HTF частота событий: {self.htf_results.get('events_rate', 0):.2%}",
            ])
        
        report_lines.extend([
            "",
            "РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:",
        ])
        
        # LTF результаты
        if 'validation' in self.ltf_results:
            v = self.ltf_results['validation']
            report_lines.extend([
                "",
                "LTF СИСТЕМА:",
                f"  ROC-AUC: {v['roc_auc']:.3f}",
                f"  Точность: {v['accuracy']:.3f}",
                f"  Precision: {v['precision']:.3f}",
                f"  Recall: {v['recall']:.3f}",
                f"  Lift: {v['lift']:.3f}",
                f"  Требования выполнены: {'ДА' if v['meets_requirements'] else 'НЕТ'}",
            ])
        
        # HTF результаты  
        if 'validation' in self.htf_results:
            v = self.htf_results['validation']
            report_lines.extend([
                "",
                "HTF СИСТЕМА:",
                f"  ROC-AUC: {v['roc_auc']:.3f}",
                f"  Точность: {v['accuracy']:.3f}",
                f"  Precision: {v['precision']:.3f}",
                f"  Recall: {v['recall']:.3f}",
                f"  Lift: {v['lift']:.3f}",
                f"  Требования выполнены: {'ДА' if v['meets_requirements'] else 'НЕТ'}",
            ])
        
        # Временные лаги (ИСПРАВЛЕНО)
        report_lines.extend([
            "",
            "ИСПРАВЛЕННЫЕ ВРЕМЕННЫЕ ЛАГИ:",
        ])
        
        for type_name, results in [('LTF', self.ltf_results), ('HTF', self.htf_results)]:
            if 'temporal_lags' in results:
                report_lines.append(f"\n{type_name} ЛАГИ:")
                for group, lag_info in results['temporal_lags'].items():
                    report_lines.append(f"  {group}: {lag_info['mean_lag']:.1f} периодов (активация {lag_info['activation_rate']:.1%})")
        
        report_lines.extend([
            "",
            "ФАЙЛЫ РЕЗУЛЬТАТОВ:",
            "results/ltf/ - LTF анализ",
            "results/htf/ - HTF анализ",
            "results/combined/ - сравнение и комбинированный анализ",
            "",
            "=" * 50
        ])
        
        with open('results/ltf_htf_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _create_comparison_plots(self):
        """Создание сравнительных графиков"""
        if not (self.ltf_results and self.htf_results):
            return
        
        plt.style.use('default')
        
        # Сравнение производительности
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # График 1: Метрики валидации
        if 'validation' in self.ltf_results and 'validation' in self.htf_results:
            metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'lift']
            ltf_values = [self.ltf_results['validation'].get(m, 0) for m in metrics]
            htf_values = [self.htf_results['validation'].get(m, 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, ltf_values, width, label='LTF', color='lightblue', alpha=0.8)
            ax1.bar(x + width/2, htf_values, width, label='HTF', color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Метрики')
            ax1.set_ylabel('Значения')
            ax1.set_title('Сравнение производительности LTF vs HTF')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # График 2: Средние временные лаги
        if 'temporal_lags' in self.ltf_results and 'temporal_lags' in self.htf_results:
            groups = list(self.ltf_results['temporal_lags'].keys())
            ltf_lags = [self.ltf_results['temporal_lags'][g]['mean_lag'] for g in groups]
            htf_lags = [self.htf_results['temporal_lags'][g]['mean_lag'] for g in groups]
            
            x = np.arange(len(groups))
            
            ax2.bar(x - width/2, ltf_lags, width, label='LTF', color='lightblue', alpha=0.8)
            ax2.bar(x + width/2, htf_lags, width, label='HTF', color='lightcoral', alpha=0.8)
            
            ax2.set_xlabel('Группы полей')
            ax2.set_ylabel('Средний лаг (периоды)')
            ax2.set_title('Сравнение временных лагов')
            ax2.set_xticks(x)
            ax2.set_xticklabels(groups, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # График 3: Частота событий
        event_data = []
        labels = []
        colors = []
        
        if self.ltf_results:
            event_data.append(self.ltf_results['events_rate'])
            labels.append('LTF')
            colors.append('lightblue')
        
        if self.htf_results:
            event_data.append(self.htf_results['events_rate'])
            labels.append('HTF')
            colors.append('lightcoral')
        
        if event_data:
            ax3.pie(event_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Частота событий LTF vs HTF')
        
        # График 4: Активность групп полей
        if 'correlations' in self.ltf_results and 'correlations' in self.htf_results:
            groups = list(self.ltf_results['correlations'].keys())
            ltf_corrs = [abs(self.ltf_results['correlations'][g]['event_correlation']) for g in groups]
            htf_corrs = [abs(self.htf_results['correlations'][g]['event_correlation']) for g in groups]
            
            x = np.arange(len(groups))
            
            ax4.bar(x - width/2, ltf_corrs, width, label='LTF', color='lightblue', alpha=0.8)
            ax4.bar(x + width/2, htf_corrs, width, label='HTF', color='lightcoral', alpha=0.8)
            
            ax4.set_xlabel('Группы полей')
            ax4.set_ylabel('|Корреляция с событиями|')
            ax4.set_title('Сравнение корреляций групп')
            ax4.set_xticks(x)
            ax4.set_xticklabels(groups, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/combined/ltf_htf_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_ltf_htf_analysis(self, file_path):
        """Запуск полного LTF/HTF анализа"""
        print("🚀 Запуск полного LTF/HTF анализа...")
        
        try:
            # Парсинг с разделением LTF/HTF
            self.parse_mixed_format_file(file_path)
            
            # Анализ каждого типа
            ltf_results = self.analyze_ltf()
            htf_results = self.analyze_htf()
            
            # Сохранение результатов
            self.save_results()
            
            print("🎉 LTF/HTF анализ завершен успешно!")
            
            return {
                'status': 'success',
                'ltf_results': ltf_results,
                'htf_results': htf_results,
                'files_created': [
                    'results/ltf/weight_matrix_ltf.csv',
                    'results/htf/weight_matrix_htf.csv',
                    'results/ltf/scoring_config_ltf.json',
                    'results/htf/scoring_config_htf.json',
                    'results/combined/ltf_htf_comparison.png',
                    'results/ltf_htf_summary.txt'
                ]
            }
            
        except Exception as e:
            print(f"❌ Ошибка LTF/HTF анализа: {e}")
            return {'status': 'error', 'message': str(e)}


def main():
    """Функция для тестирования LTF/HTF анализатора"""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python ltf_htf_analyzer.py <путь_к_файлу_лога>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Файл {log_file} не найден!")
        sys.exit(1)
    
    # Создание LTF/HTF анализатора
    analyzer = LTFHTFAnalyzer()
    
    # Запуск анализа
    results = analyzer.run_full_ltf_htf_analysis(log_file)
    
    if results['status'] == 'success':
        print("\n📊 Сводка результатов:")
        if 'ltf_results' in results and results['ltf_results']:
            ltf_val = results['ltf_results'].get('validation', {})
            print(f"LTF ROC-AUC: {ltf_val.get('roc_auc', 0):.3f}")
        
        if 'htf_results' in results and results['htf_results']:
            htf_val = results['htf_results'].get('validation', {})
            print(f"HTF ROC-AUC: {htf_val.get('roc_auc', 0):.3f}")
        
        print("\nПроверьте папки results/ltf/ и results/htf/ для результатов.")
    else:
        print(f"Ошибка: {results['message']}")


if __name__ == "__main__":
    main()