#!/usr/bin/env python3
"""
Модуль продвинутого определения событий
Новые типы: откаты 2-3%, 3-5%, 5-7%, 7-10%, 10%+, кульминации, консолидации

Data-driven подход без априорных знаний
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from scipy.signal import argrelextrema


class AdvancedEventDetector:
    """
    Продвинутый детектор событий с новыми типами
    
    Типы событий:
    1. Откаты: 2-3%, 3-5%, 5-7%, 7-10%, 10%+ (без обновления лоя)
    2. Кульминации: долгосрочные экстремумы с разворотом
    3. Продолжение/Развитие: пробои уровней, развитие тренда  
    4. Консолидации: боковые движения, флеты
    5. Переходные зоны: области между активными фазами
    """
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация детектора событий"""
        self.config = self._load_config(config_path)
        self.data = None
        self.events = None
        self.extrema = None
        self.event_stats = {}
        
        # Параметры из конфигурации
        self.retracement_levels = self.config.get('advanced_events', {}).get('retracement_levels', [2, 3, 5, 7, 10])
        self.retracement_time_window = self.config.get('advanced_events', {}).get('retracement_time_window', [1, 90])
        self.min_extremum_move = self.config.get('advanced_events', {}).get('min_extremum_move', 1.0)
        self.culmination_threshold = self.config.get('advanced_events', {}).get('culmination_threshold', 0.8)
        self.consolidation_volatility_threshold = self.config.get('advanced_events', {}).get('consolidation_volatility_threshold', 0.5)
        
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        default_config = {
            'advanced_events': {
                'retracement_levels': [2, 3, 5, 7, 10],
                'retracement_time_window': [1, 90],
                'min_extremum_move': 1.0,
                'culmination_threshold': 0.8,
                'consolidation_volatility_threshold': 0.5
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def detect_advanced_events(self, data):
        """
        Главная функция определения продвинутых событий
        
        Args:
            data: DataFrame с OHLC данными
            
        Returns:
            DataFrame с дополнительными колонками событий
        """
        print("🎯 Определение продвинутых типов событий...")
        
        self.data = data.copy()
        
        # Проверка обязательных колонок
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.data.columns:
                print(f"⚠️ Отсутствует колонка {col}")
                return self.data
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(method='ffill')
        
        # Расчет базовых метрик
        self._calculate_base_metrics()
        
        # Определение экстремумов
        self._find_extrema()
        
        # Определение различных типов событий
        self._detect_retracements()
        self._detect_culminations()
        self._detect_continuations()
        self._detect_consolidations()
        self._detect_transition_zones()
        
        # Расчет статистики событий
        self._calculate_event_statistics()
        
        print(f"✅ Продвинутое определение событий завершено")
        self._print_event_summary()
        
        return self.data
    
    def _calculate_base_metrics(self):
        """Расчет базовых метрик для анализа"""
        
        # Ценовые метрики
        self.data['hl_range'] = self.data['high'] - self.data['low']
        self.data['true_range'] = self._calculate_true_range()
        self.data['price_change_pct'] = ((self.data['close'] - self.data['open']) / self.data['open']) * 100
        self.data['price_change_abs'] = abs(self.data['price_change_pct'])
        
        # Волатильность (rolling)
        for window in [5, 10, 20]:
            self.data[f'volatility_{window}'] = self.data['true_range'].rolling(window=window).std()
            self.data[f'price_range_{window}'] = self.data['hl_range'].rolling(window=window).mean()
        
        # Momentum индикаторы
        for window in [3, 5, 10, 20]:
            self.data[f'momentum_{window}'] = self.data['close'].pct_change(window) * 100
            self.data[f'roc_{window}'] = ((self.data['close'] / self.data['close'].shift(window)) - 1) * 100
        
        # Скользящие средние для трендов
        for window in [10, 20, 50]:
            self.data[f'sma_{window}'] = self.data['close'].rolling(window=window).mean()
            self.data[f'close_vs_sma_{window}'] = ((self.data['close'] / self.data[f'sma_{window}']) - 1) * 100
        
        # Обработка NaN значений
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(0, inplace=True)
        
        print(f"   Рассчитаны базовые метрики")
    
    def _calculate_true_range(self):
        """Расчет True Range"""
        high_low = self.data['high'] - self.data['low']
        high_close_prev = abs(self.data['high'] - self.data['close'].shift(1))
        low_close_prev = abs(self.data['low'] - self.data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.fillna(high_low)
    
    def _find_extrema(self):
        """Определение локальных экстремумов (пиков и впадин)"""
        
        # Используем различные окна для поиска экстремумов
        windows = [5, 10, 15, 20]
        
        extrema_data = []
        
        for window in windows:
            # Поиск локальных максимумов
            highs = argrelextrema(self.data['high'].values, np.greater, order=window)[0]
            for idx in highs:
                if idx < len(self.data):
                    extrema_data.append({
                        'index': idx,
                        'type': 'high',
                        'price': self.data.iloc[idx]['high'],
                        'window': window,
                        'timestamp': self.data.index[idx] if hasattr(self.data.index, '__getitem__') else idx
                    })
            
            # Поиск локальных минимумов
            lows = argrelextrema(self.data['low'].values, np.less, order=window)[0]
            for idx in lows:
                if idx < len(self.data):
                    extrema_data.append({
                        'index': idx,
                        'type': 'low',
                        'price': self.data.iloc[idx]['low'],
                        'window': window,
                        'timestamp': self.data.index[idx] if hasattr(self.data.index, '__getitem__') else idx
                    })
        
        # Создание DataFrame экстремумов
        if extrema_data:
            self.extrema = pd.DataFrame(extrema_data).drop_duplicates(['index', 'type']).sort_values('index')
            
            # Фильтрация значимых экстремумов
            self._filter_significant_extrema()
            
            print(f"   Найдено {len(self.extrema)} экстремумов")
        else:
            self.extrema = pd.DataFrame()
            print("   Экстремумы не найдены")
    
    def _filter_significant_extrema(self):
        """Фильтрация значимых экстремумов"""
        if self.extrema.empty:
            return
        
        # Расчет значимости движения от предыдущего экстремума
        significant_extrema = []
        
        for i, extremum in self.extrema.iterrows():
            if len(significant_extrema) == 0:
                significant_extrema.append(extremum)
                continue
            
            prev_extremum = significant_extrema[-1]
            
            # Расчет процентного изменения
            price_change = abs((extremum['price'] - prev_extremum['price']) / prev_extremum['price']) * 100
            
            # Проверка значимости
            if price_change >= self.min_extremum_move:
                significant_extrema.append(extremum)
        
        self.extrema = pd.DataFrame(significant_extrema)
    
    def _detect_retracements(self):
        """Определение откатов различных уровней"""
        
        # Инициализация колонок для откатов
        for level in self.retracement_levels:
            if level < 10:
                self.data[f'retracement_{level}_{level+1}pct'] = 0
            else:
                self.data[f'retracement_{level}pct_plus'] = 0
        
        self.data['retracement_type'] = 'none'
        self.data['retracement_strength'] = 0.0
        
        if self.extrema.empty:
            print("   Откаты: экстремумы не найдены")
            return
        
        print(f"   Анализ откатов для {len(self.extrema)} экстремумов...")
        
        retracement_count = 0
        
        # Анализ каждого экстремума на предмет откатов
        for i, extremum in self.extrema.iterrows():
            extremum_idx = extremum['index']
            extremum_price = extremum['price']
            extremum_type = extremum['type']
            
            # Поиск откатов после экстремума
            retracement_info = self._analyze_retracement_from_extremum(
                extremum_idx, extremum_price, extremum_type
            )
            
            if retracement_info:
                retracement_count += 1
                
                # Маркировка события в данных
                retracement_level = retracement_info['level']
                retracement_end_idx = retracement_info['end_index']
                
                # Определение типа отката по уровню
                level_column = self._get_retracement_column(retracement_level)
                
                if level_column in self.data.columns:
                    self.data.loc[retracement_end_idx, level_column] = 1
                    self.data.loc[retracement_end_idx, 'retracement_type'] = f"retracement_{retracement_level:.1f}pct"
                    self.data.loc[retracement_end_idx, 'retracement_strength'] = retracement_info['strength']
        
        print(f"   Найдено {retracement_count} откатов")
    
    def _analyze_retracement_from_extremum(self, extremum_idx, extremum_price, extremum_type):
        """Анализ отката от конкретного экстремума"""
        
        # Определяем диапазон для поиска отката
        min_time, max_time = self.retracement_time_window
        start_search = extremum_idx + 1
        end_search = min(len(self.data), extremum_idx + max_time)
        
        if start_search >= len(self.data):
            return None
        
        search_data = self.data.iloc[start_search:end_search]
        
        max_retracement = 0
        max_retracement_idx = None
        retracement_end_idx = None
        
        current_extremum_price = extremum_price
        
        for idx, row in search_data.iterrows():
            actual_idx = start_search + (idx - search_data.index[0]) if hasattr(search_data.index, '__getitem__') else idx
            
            if extremum_type == 'high':
                # Для максимума ищем движение вниз
                retracement_pct = ((extremum_price - row['low']) / extremum_price) * 100
                
                # Проверяем, что не обновился лой
                if row['low'] <= current_extremum_price:
                    current_extremum_price = row['low']
                    
                    if retracement_pct > max_retracement:
                        max_retracement = retracement_pct
                        max_retracement_idx = actual_idx
                else:
                    # Если цена пошла вверх без обновления лоя - возможный конец отката
                    if max_retracement > 0:
                        retracement_end_idx = actual_idx
                        break
                        
            else:  # extremum_type == 'low'
                # Для минимума ищем движение вверх
                retracement_pct = ((row['high'] - extremum_price) / extremum_price) * 100
                
                # Проверяем, что не обновился хай
                if row['high'] >= current_extremum_price:
                    current_extremum_price = row['high']
                    
                    if retracement_pct > max_retracement:
                        max_retracement = retracement_pct
                        max_retracement_idx = actual_idx
                else:
                    # Если цена пошла вниз без обновления хая - возможный конец отката
                    if max_retracement > 0:
                        retracement_end_idx = actual_idx
                        break
        
        # Проверяем соответствие уровням откатов
        if max_retracement >= 2.0:  # Минимальный откат 2%
            retracement_level = self._classify_retracement_level(max_retracement)
            
            return {
                'level': retracement_level,
                'strength': max_retracement,
                'max_index': max_retracement_idx,
                'end_index': retracement_end_idx or max_retracement_idx,
                'extremum_type': extremum_type
            }
        
        return None
    
    def _classify_retracement_level(self, retracement_pct):
        """Классификация уровня отката"""
        if retracement_pct >= 10:
            return 10  # 10%+
        elif retracement_pct >= 7:
            return 7   # 7-10%
        elif retracement_pct >= 5:
            return 5   # 5-7%
        elif retracement_pct >= 3:
            return 3   # 3-5%
        else:
            return 2   # 2-3%
    
    def _get_retracement_column(self, level):
        """Получение названия колонки для уровня отката"""
        if level >= 10:
            return 'retracement_10pct_plus'
        elif level >= 7:
            return 'retracement_7_10pct'
        elif level >= 5:
            return 'retracement_5_7pct'
        elif level >= 3:
            return 'retracement_3_5pct'
        else:
            return 'retracement_2_3pct'
    
    def _detect_culminations(self):
        """Определение кульминаций (долгосрочных экстремумов с разворотом)"""
        
        self.data['culmination'] = 0
        self.data['culmination_strength'] = 0.0
        self.data['culmination_type'] = 'none'
        
        if self.extrema.empty:
            print("   Кульминации: экстремумы не найдены")
            return
        
        print(f"   Анализ кульминаций...")
        
        culmination_count = 0
        
        # Поиск кульминаций среди экстремумов
        for i, extremum in self.extrema.iterrows():
            extremum_idx = extremum['index']
            
            # Анализ долгосрочного разворота после экстремума
            culmination_info = self._analyze_culmination(extremum_idx, extremum)
            
            if culmination_info and culmination_info['strength'] >= self.culmination_threshold:
                culmination_count += 1
                
                self.data.loc[extremum_idx, 'culmination'] = 1
                self.data.loc[extremum_idx, 'culmination_strength'] = culmination_info['strength']
                self.data.loc[extremum_idx, 'culmination_type'] = culmination_info['type']
        
        print(f"   Найдено {culmination_count} кульминаций")
    
    def _analyze_culmination(self, extremum_idx, extremum):
        """Анализ кульминации от экстремума"""
        
        # Поиск долгосрочного движения после экстремума
        lookforward = min(50, len(self.data) - extremum_idx - 1)  # До 50 периодов вперед
        
        if lookforward < 10:  # Минимальное окно для анализа
            return None
        
        extremum_price = extremum['price']
        extremum_type = extremum['type']
        
        future_data = self.data.iloc[extremum_idx + 1:extremum_idx + 1 + lookforward]
        
        if len(future_data) == 0:
            return None
        
        if extremum_type == 'high':
            # Ищем устойчивое движение вниз
            min_price = future_data['low'].min()
            decline_pct = ((extremum_price - min_price) / extremum_price) * 100
            
            # Проверяем устойчивость движения
            stability_score = self._calculate_movement_stability(future_data, 'down')
            
            culmination_strength = (decline_pct / 10) * stability_score  # Нормализация
            
            if decline_pct >= 5.0 and stability_score > 0.6:  # Минимальные критерии
                return {
                    'strength': min(culmination_strength, 1.0),
                    'type': 'top_culmination',
                    'decline_pct': decline_pct,
                    'stability': stability_score
                }
        
        else:  # extremum_type == 'low'
            # Ищем устойчивое движение вверх
            max_price = future_data['high'].max()
            rise_pct = ((max_price - extremum_price) / extremum_price) * 100
            
            # Проверяем устойчивость движения
            stability_score = self._calculate_movement_stability(future_data, 'up')
            
            culmination_strength = (rise_pct / 10) * stability_score  # Нормализация
            
            if rise_pct >= 5.0 and stability_score > 0.6:  # Минимальные критерии
                return {
                    'strength': min(culmination_strength, 1.0),
                    'type': 'bottom_culmination',
                    'rise_pct': rise_pct,
                    'stability': stability_score
                }
        
        return None
    
    def _calculate_movement_stability(self, data, direction):
        """Расчет стабильности движения"""
        if len(data) < 5:
            return 0
        
        if direction == 'down':
            # Для движения вниз: процент периодов с понижением
            down_periods = (data['close'].diff() < 0).sum()
            stability = down_periods / len(data)
        else:
            # Для движения вверх: процент периодов с повышением
            up_periods = (data['close'].diff() > 0).sum()
            stability = up_periods / len(data)
        
        return stability
    
    def _detect_continuations(self):
        """Определение продолжений/развития движения"""
        
        self.data['continuation'] = 0
        self.data['continuation_strength'] = 0.0
        self.data['continuation_type'] = 'none'
        
        print(f"   Анализ продолжений движения...")
        
        continuation_count = 0
        
        # Анализ пробоев уровней и продолжений трендов
        for i in range(20, len(self.data) - 10):  # Нужна история и будущее для анализа
            
            continuation_info = self._analyze_continuation(i)
            
            if continuation_info:
                continuation_count += 1
                
                self.data.iloc[i, self.data.columns.get_loc('continuation')] = 1
                self.data.iloc[i, self.data.columns.get_loc('continuation_strength')] = continuation_info['strength']
                self.data.iloc[i, self.data.columns.get_loc('continuation_type')] = continuation_info['type']
        
        print(f"   Найдено {continuation_count} продолжений")
    
    def _analyze_continuation(self, idx):
        """Анализ продолжения движения в точке"""
        
        # Анализ истории (20 периодов назад)
        history_start = max(0, idx - 20)
        history_data = self.data.iloc[history_start:idx]
        
        # Анализ будущего (10 периодов вперед)
        future_end = min(len(self.data), idx + 10)
        future_data = self.data.iloc[idx:future_end]
        
        if len(history_data) < 10 or len(future_data) < 5:
            return None
        
        current_price = self.data.iloc[idx]['close']
        
        # Определение тренда в истории
        history_trend = self._identify_trend(history_data)
        
        # Проверка пробоя уровней
        resistance_level = history_data['high'].max()
        support_level = history_data['low'].min()
        
        # Анализ развития в будущем
        future_movement = self._analyze_future_movement(future_data, current_price)
        
        # Определение продолжения
        if history_trend == 'up' and current_price > resistance_level:
            # Пробой сопротивления при восходящем тренде
            if future_movement['direction'] == 'up' and future_movement['strength'] > 0.5:
                return {
                    'strength': future_movement['strength'],
                    'type': 'uptrend_continuation',
                    'breakout_level': resistance_level
                }
        
        elif history_trend == 'down' and current_price < support_level:
            # Пробой поддержки при нисходящем тренде
            if future_movement['direction'] == 'down' and future_movement['strength'] > 0.5:
                return {
                    'strength': future_movement['strength'],
                    'type': 'downtrend_continuation',
                    'breakout_level': support_level
                }
        
        return None
    
    def _identify_trend(self, data):
        """Определение тренда в данных"""
        if len(data) < 5:
            return 'sideways'
        
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        
        price_change_pct = ((end_price - start_price) / start_price) * 100
        
        # Простая классификация тренда
        if price_change_pct > 2:
            return 'up'
        elif price_change_pct < -2:
            return 'down'
        else:
            return 'sideways'
    
    def _analyze_future_movement(self, future_data, current_price):
        """Анализ движения в будущем"""
        if len(future_data) < 3:
            return {'direction': 'none', 'strength': 0}
        
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        
        upward_move = ((max_price - current_price) / current_price) * 100
        downward_move = ((current_price - min_price) / current_price) * 100
        
        if upward_move > downward_move and upward_move > 1:
            return {'direction': 'up', 'strength': min(upward_move / 5, 1.0)}
        elif downward_move > upward_move and downward_move > 1:
            return {'direction': 'down', 'strength': min(downward_move / 5, 1.0)}
        else:
            return {'direction': 'sideways', 'strength': 0.1}
    
    def _detect_consolidations(self):
        """Определение консолидаций (боковые движения)"""
        
        self.data['consolidation'] = 0
        self.data['consolidation_strength'] = 0.0
        
        print(f"   Анализ консолидаций...")
        
        consolidation_count = 0
        window = 10  # Окно для анализа консолидации
        
        for i in range(window, len(self.data) - window):
            consolidation_info = self._analyze_consolidation(i, window)
            
            if consolidation_info:
                consolidation_count += 1
                
                # Маркировка всего периода консолидации
                start_idx = max(0, i - window // 2)
                end_idx = min(len(self.data), i + window // 2)
                
                for idx in range(start_idx, end_idx):
                    self.data.iloc[idx, self.data.columns.get_loc('consolidation')] = 1
                    self.data.iloc[idx, self.data.columns.get_loc('consolidation_strength')] = consolidation_info['strength']
        
        print(f"   Найдено {consolidation_count} зон консолидации")
    
    def _analyze_consolidation(self, idx, window):
        """Анализ консолидации в окне"""
        
        start_idx = max(0, idx - window // 2)
        end_idx = min(len(self.data), idx + window // 2)
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        if len(window_data) < window * 0.8:  # Минимальное количество данных
            return None
        
        # Расчет характеристик консолидации
        price_range = window_data['high'].max() - window_data['low'].min()
        avg_price = window_data['close'].mean()
        range_pct = (price_range / avg_price) * 100
        
        # Волатильность в окне
        volatility = window_data['true_range'].mean() / avg_price * 100
        
        # Критерии консолидации
        if (range_pct < 3.0 and  # Диапазон менее 3%
            volatility < self.consolidation_volatility_threshold):  # Низкая волатильность
            
            return {
                'strength': max(0.1, 1.0 - (range_pct / 3.0)),  # Обратная зависимость от диапазона
                'range_pct': range_pct,
                'volatility': volatility
            }
        
        return None
    
    def _detect_transition_zones(self):
        """Определение переходных зон"""
        
        self.data['transition_zone'] = 0
        self.data['transition_strength'] = 0.0
        
        print(f"   Анализ переходных зон...")
        
        transition_count = 0
        
        # Поиск зон между активными фазами
        for i in range(20, len(self.data) - 20):
            transition_info = self._analyze_transition_zone(i)
            
            if transition_info:
                transition_count += 1
                
                self.data.iloc[i, self.data.columns.get_loc('transition_zone')] = 1
                self.data.iloc[i, self.data.columns.get_loc('transition_strength')] = transition_info['strength']
        
        print(f"   Найдено {transition_count} переходных зон")
    
    def _analyze_transition_zone(self, idx):
        """Анализ переходной зоны"""
        
        # Анализируем окружение для поиска изменений режима
        before_window = self.data.iloc[idx-20:idx]
        after_window = self.data.iloc[idx:idx+20]
        
        if len(before_window) < 15 or len(after_window) < 15:
            return None
        
        # Характеристики до и после
        before_volatility = before_window['volatility_5'].mean()
        after_volatility = after_window['volatility_5'].mean()
        
        before_trend = self._identify_trend(before_window)
        after_trend = self._identify_trend(after_window)
        
        # Критерий переходной зоны: изменение режима
        if (before_trend != after_trend and 
            abs(before_volatility - after_volatility) > 0.1):
            
            transition_strength = min(1.0, abs(before_volatility - after_volatility))
            
            return {
                'strength': transition_strength,
                'before_trend': before_trend,
                'after_trend': after_trend,
                'volatility_change': abs(before_volatility - after_volatility)
            }
        
        return None
    
    def _calculate_event_statistics(self):
        """Расчет статистики по всем типам событий"""
        
        # Подсчет событий каждого типа
        event_types = [
            'retracement_2_3pct', 'retracement_3_5pct', 'retracement_5_7pct', 
            'retracement_7_10pct', 'retracement_10pct_plus',
            'culmination', 'continuation', 'consolidation', 'transition_zone'
        ]
        
        self.event_stats = {}
        
        for event_type in event_types:
            if event_type in self.data.columns:
                count = self.data[event_type].sum()
                rate = count / len(self.data)
                
                self.event_stats[event_type] = {
                    'count': int(count),
                    'rate': float(rate),
                    'percentage': float(rate * 100)
                }
        
        # Общая статистика
        total_events = sum(stats['count'] for stats in self.event_stats.values())
        
        self.event_stats['summary'] = {
            'total_events': total_events,
            'total_records': len(self.data),
            'overall_event_rate': total_events / len(self.data) if len(self.data) > 0 else 0
        }
    
    def _print_event_summary(self):
        """Вывод сводки по событиям"""
        print("\n📊 СВОДКА ПО ПРОДВИНУТЫМ СОБЫТИЯМ:")
        
        if not self.event_stats:
            print("   Статистика не рассчитана")
            return
        
        for event_type, stats in self.event_stats.items():
            if event_type != 'summary':
                print(f"   {event_type}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        summary = self.event_stats.get('summary', {})
        print(f"\n   Всего событий: {summary.get('total_events', 0)}")
        print(f"   Общая частота: {summary.get('overall_event_rate', 0):.2%}")
    
    def save_event_analysis(self, output_dir="results/advanced_events"):
        """Сохранение результатов анализа событий"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение данных с событиями
        self.data.to_csv(output_path / "advanced_events_data.csv", index=False)
        
        # Сохранение статистики
        with open(output_path / "event_statistics.json", 'w') as f:
            json.dump(self.event_stats, f, indent=2)
        
        # Сохранение экстремумов
        if not self.extrema.empty:
            self.extrema.to_csv(output_path / "extrema_analysis.csv", index=False)
        
        # Создание отчета
        self._create_event_report(output_path)
        
        # Создание визуализаций
        self._create_event_visualizations(output_path)
        
        print(f"✅ Результаты анализа событий сохранены в {output_path}")
    
    def _create_event_report(self, output_path):
        """Создание текстового отчета"""
        
        report_lines = [
            "ПРОДВИНУТЫЙ АНАЛИЗ СОБЫТИЙ - ОТЧЕТ",
            "=" * 50,
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Записей проанализировано: {len(self.data)}",
            "",
            "ПАРАМЕТРЫ АНАЛИЗА:",
            f"Уровни откатов: {self.retracement_levels}%",
            f"Временное окно откатов: {self.retracement_time_window[0]}-{self.retracement_time_window[1]} мин",
            f"Минимальное движение экстремума: {self.min_extremum_move}%",
            f"Порог кульминации: {self.culmination_threshold}",
            f"Порог консолидации: {self.consolidation_volatility_threshold}",
            "",
            "РЕЗУЛЬТАТЫ АНАЛИЗА:",
        ]
        
        if self.event_stats:
            for event_type, stats in self.event_stats.items():
                if event_type != 'summary':
                    report_lines.append(f"{event_type}: {stats['count']} событий ({stats['percentage']:.1f}%)")
            
            summary = self.event_stats.get('summary', {})
            report_lines.extend([
                "",
                f"ИТОГО: {summary.get('total_events', 0)} событий",
                f"Общая частота событий: {summary.get('overall_event_rate', 0):.2%}",
            ])
        
        if not self.extrema.empty:
            report_lines.extend([
                "",
                f"ЭКСТРЕМУМЫ: {len(self.extrema)} найдено",
                f"Максимумы: {len(self.extrema[self.extrema['type'] == 'high'])}",
                f"Минимумы: {len(self.extrema[self.extrema['type'] == 'low'])}",
            ])
        
        report_lines.extend([
            "",
            "ФАЙЛЫ РЕЗУЛЬТАТОВ:",
            "advanced_events_data.csv - данные с маркированными событиями",
            "event_statistics.json - детальная статистика",
            "extrema_analysis.csv - анализ экстремумов",
            "event_timeline.png - временная линия событий",
            "",
            "=" * 50
        ])
        
        with open(output_path / "event_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _create_event_visualizations(self, output_path):
        """Создание визуализаций событий"""
        
        plt.style.use('default')
        
        # График 1: Временная линия событий
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Ценовой график с событиями
        ax1.plot(self.data.index, self.data['close'], label='Close Price', linewidth=1)
        
        # Маркировка различных типов событий
        event_colors = {
            'retracement_2_3pct': 'lightblue',
            'retracement_3_5pct': 'blue', 
            'retracement_5_7pct': 'darkblue',
            'retracement_7_10pct': 'purple',
            'retracement_10pct_plus': 'red',
            'culmination': 'orange',
            'continuation': 'green',
            'consolidation': 'gray',
            'transition_zone': 'yellow'
        }
        
        for event_type, color in event_colors.items():
            if event_type in self.data.columns:
                event_indices = self.data[self.data[event_type] == 1].index
                if len(event_indices) > 0:
                    ax1.scatter(event_indices, self.data.loc[event_indices, 'close'], 
                              c=color, label=event_type, s=30, alpha=0.7)
        
        ax1.set_title('Временная линия продвинутых событий')
        ax1.set_ylabel('Цена')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Статистика событий
        if self.event_stats:
            event_names = [name for name in self.event_stats.keys() if name != 'summary']
            event_counts = [self.event_stats[name]['count'] for name in event_names]
            
            bars = ax2.bar(range(len(event_names)), event_counts, color=plt.cm.tab10(range(len(event_names))))
            ax2.set_xlabel('Типы событий')
            ax2.set_ylabel('Количество')
            ax2.set_title('Статистика по типам событий')
            ax2.set_xticks(range(len(event_names)))
            ax2.set_xticklabels(event_names, rotation=45, ha='right')
            
            # Добавление значений на столбцы
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / "event_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # График экстремумов если есть
        if not self.extrema.empty:
            self._plot_extrema_analysis(output_path)
    
    def _plot_extrema_analysis(self, output_path):
        """График анализа экстремумов"""
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Ценовой график
        ax.plot(self.data.index, self.data['close'], label='Close Price', linewidth=1, color='black')
        
        # Маркировка экстремумов
        for _, extremum in self.extrema.iterrows():
            idx = extremum['index']
            price = extremum['price']
            ext_type = extremum['type']
            
            if idx < len(self.data):
                color = 'red' if ext_type == 'high' else 'green'
                marker = 'v' if ext_type == 'high' else '^'
                
                ax.scatter(idx, price, c=color, marker=marker, s=100, 
                          label=f'{ext_type.capitalize()}' if ext_type not in ax.get_legend_handles_labels()[1] else "")
        
        ax.set_title('Анализ локальных экстремумов')
        ax.set_xlabel('Индекс')
        ax.set_ylabel('Цена')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "extrema_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Функция для тестирования детектора событий"""
    import sys
    
    # Простой тест на синтетических данных
    print("🧪 Тестирование детектора продвинутых событий...")
    
    # Создание тестовых данных
    np.random.seed(42)
    n_points = 200
    
    # Генерация ценовых данных с трендом и волатильностью
    base_price = 50000
    trend = np.cumsum(np.random.normal(0, 0.5, n_points))
    noise = np.random.normal(0, 100, n_points)
    
    close_prices = base_price + trend * 50 + noise
    
    # Добавление искусственных экстремумов
    close_prices[50] *= 1.05  # Пик
    close_prices[100] *= 0.95  # Впадина
    close_prices[150] *= 1.08  # Большой пик
    
    # OHLC данные
    test_data = pd.DataFrame({
        'open': close_prices + np.random.normal(0, 10, n_points),
        'high': close_prices + abs(np.random.normal(20, 10, n_points)),
        'low': close_prices - abs(np.random.normal(20, 10, n_points)),
        'close': close_prices
    })
    
    # Тестирование детектора
    detector = AdvancedEventDetector()
    result_data = detector.detect_advanced_events(test_data)
    
    # Сохранение результатов
    detector.save_event_analysis("results/test_advanced_events")
    
    print("✅ Тестирование завершено. Результаты в results/test_advanced_events/")


if __name__ == "__main__":
    main()