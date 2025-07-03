#!/usr/bin/env python3
"""
Система стоп-полей и VETO логики
Блокирует ложные сигналы и фильтрует конфликтующие активации

Data-driven подход: автоматическое определение блокирующих полей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score


class VetoSystem:
    """
    Система стоп-полей и VETO логики
    
    Задачи:
    1. Поиск полей-блокираторов (стоп-поля)
    2. Определение конфликтующих сигналов
    3. Фильтрация ложных активаций
    4. Система "разрешений" на вход в сделку
    """
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация VETO системы"""
        self.config = self._load_config(config_path)
        
        # Данные и результаты
        self.data = None
        self.features = None
        self.targets = None
        self.veto_rules = {}
        self.blocking_fields = {}
        self.conflict_patterns = {}
        self.false_signal_analysis = {}
        
        # Параметры из конфигурации
        veto_config = self.config.get('veto_system', {})
        self.enable_blocking = veto_config.get('enable_blocking', True)
        self.veto_thresholds = veto_config.get('veto_thresholds', {})
        self.min_confirming_signals = veto_config.get('min_confirming_signals', 2)
        
        # Пороги по умолчанию
        self.high_volatility_threshold = self.veto_thresholds.get('high_volatility', 3.0)
        self.conflicting_signals_threshold = self.veto_thresholds.get('conflicting_signals', 0.7)
        self.low_confidence_threshold = self.veto_thresholds.get('low_confidence', 0.3)
        
        # Группы полей для анализа
        self.field_groups = {
            'group_1': ['rd', 'md', 'cd', 'cmd', 'macd', 'od', 'dd', 'cvd', 'drd', 'ad', 'ed', 'hd', 'sd'],
            'group_2': ['ro', 'mo', 'co', 'cz', 'do', 'ae', 'so'],
            'group_3': ['rz', 'mz', 'ciz', 'sz', 'dz', 'cvz', 'maz', 'oz'],
            'group_4': ['ef', 'wv', 'vc', 'ze', 'nw', 'as', 'vw'],
            'group_5': ['bs', 'wa', 'pd']
        }
        
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        default_config = {
            'veto_system': {
                'enable_blocking': True,
                'veto_thresholds': {
                    'high_volatility': 3.0,
                    'conflicting_signals': 0.7,
                    'low_confidence': 0.3
                },
                'min_confirming_signals': 2
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def analyze_veto_patterns(self, data, features, targets):
        """
        Главная функция анализа VETO паттернов
        
        Args:
            data: DataFrame с исходными данными
            features: DataFrame с признаками
            targets: DataFrame с целевыми переменными
        """
        print("🚫 Анализ системы стоп-полей и VETO логики...")
        
        self.data = data.copy()
        self.features = features.copy()
        self.targets = targets.copy()
        
        # Этап 1: Поиск полей-блокираторов
        self._find_blocking_fields()
        
        # Этап 2: Анализ конфликтующих сигналов
        self._analyze_conflicting_signals()
        
        # Этап 3: Детекция ложных активаций
        self._detect_false_activations()
        
        # Этап 4: Построение VETO правил
        self._build_veto_rules()
        
        # Этап 5: Валидация системы
        self._validate_veto_system()
        
        print("✅ Анализ VETO системы завершен")
        return self.veto_rules
    
    def _find_blocking_fields(self):
        """Поиск полей-блокираторов"""
        print("🔍 Поиск полей-блокираторов...")
        
        # Стратегия: ищем поля, которые при активации снижают точность основных сигналов
        blocking_candidates = {}
        
        # Анализируем каждое поле как потенциальный блокиратор
        for feature in self.features.columns:
            try:
                blocking_analysis = self._analyze_field_as_blocker(feature)
                
                if blocking_analysis and blocking_analysis['blocking_strength'] > 0.1:
                    blocking_candidates[feature] = blocking_analysis
                    
            except Exception as e:
                continue
        
        # Сортировка по силе блокировки
        sorted_blockers = sorted(
            blocking_candidates.items(), 
            key=lambda x: x[1]['blocking_strength'], 
            reverse=True
        )
        
        # Отбор топ блокираторов
        self.blocking_fields = dict(sorted_blockers[:20])  # Топ 20 блокираторов
        
        print(f"   Найдено {len(self.blocking_fields)} полей-блокираторов")
        
        # Детальный анализ топ блокираторов
        for field, analysis in list(self.blocking_fields.items())[:5]:
            print(f"     {field}: блокировка {analysis['blocking_strength']:.3f}, ложные сигналы {analysis['false_positive_rate']:.1%}")
    
    def _analyze_field_as_blocker(self, field):
        """Анализ поля как потенциального блокиратора"""
        
        # Получаем активации поля
        field_data = pd.to_numeric(self.features[field], errors='coerce').fillna(0)
        
        if field_data.std() < 0.01:  # Поле без вариативности
            return None
        
        # Определяем пороги активации (различные уровни)
        thresholds = np.percentile(np.abs(field_data), [70, 80, 90, 95])
        
        best_blocking_strength = 0
        best_threshold = None
        best_analysis = None
        
        for threshold in thresholds:
            # Создаем маску активации поля
            field_active = np.abs(field_data) > threshold
            
            if field_active.sum() < 10:  # Минимальное количество активаций
                continue
            
            # Анализируем влияние на точность основных сигналов
            blocking_analysis = self._calculate_blocking_effect(field_active, field)
            
            if blocking_analysis['blocking_strength'] > best_blocking_strength:
                best_blocking_strength = blocking_analysis['blocking_strength']
                best_threshold = threshold
                best_analysis = blocking_analysis
                best_analysis['threshold'] = threshold
        
        return best_analysis
    
    def _calculate_blocking_effect(self, field_active_mask, field_name):
        """Расчет эффекта блокировки"""
        
        events = self.targets['is_event']
        
        # Разделяем данные на периоды с активацией поля и без
        active_periods = field_active_mask
        inactive_periods = ~field_active_mask
        
        # Создаем простой baseline сигнал (сумма активности других полей)
        other_fields = [col for col in self.features.columns if col != field_name]
        
        if len(other_fields) < 5:
            return {'blocking_strength': 0, 'false_positive_rate': 0}
        
        # Берем случайную выборку других полей для создания baseline
        np.random.seed(42)
        sample_fields = np.random.choice(other_fields, min(10, len(other_fields)), replace=False)
        
        baseline_signal = self.features[sample_fields].abs().sum(axis=1)
        baseline_signal = (baseline_signal > baseline_signal.quantile(0.8)).astype(int)
        
        # Точность baseline сигнала в разные периоды
        if active_periods.sum() > 10 and inactive_periods.sum() > 10:
            
            # Точность когда поле НЕ активно
            accuracy_without_field = self._calculate_accuracy(
                baseline_signal[inactive_periods], 
                events[inactive_periods]
            )
            
            # Точность когда поле активно (потенциально блокирует)
            accuracy_with_field = self._calculate_accuracy(
                baseline_signal[active_periods], 
                events[active_periods]
            )
            
            # Блокирующий эффект: насколько снижается точность при активации поля
            blocking_strength = max(0, accuracy_without_field - accuracy_with_field)
            
            # Частота ложных позитивов при активации поля
            false_positives = (baseline_signal[active_periods] == 1) & (events[active_periods] == 0)
            false_positive_rate = false_positives.sum() / max(1, active_periods.sum())
            
            return {
                'blocking_strength': blocking_strength,
                'false_positive_rate': false_positive_rate,
                'accuracy_without_field': accuracy_without_field,
                'accuracy_with_field': accuracy_with_field,
                'activations_count': active_periods.sum()
            }
        
        return {'blocking_strength': 0, 'false_positive_rate': 0}
    
    def _calculate_accuracy(self, predictions, actual):
        """Расчет точности предсказаний"""
        if len(predictions) == 0 or len(actual) == 0:
            return 0
        
        correct = (predictions == actual).sum()
        total = len(actual)
        return correct / total if total > 0 else 0
    
    def _analyze_conflicting_signals(self):
        """Анализ конфликтующих сигналов между группами"""
        print("⚔️ Анализ конфликтующих сигналов...")
        
        self.conflict_patterns = {}
        
        # Анализируем конфликты между группами полей
        for group1_name, group1_fields in self.field_groups.items():
            for group2_name, group2_fields in self.field_groups.items():
                if group1_name >= group2_name:  # Избегаем дублирования
                    continue
                
                conflict_analysis = self._analyze_group_conflicts(
                    group1_name, group1_fields, 
                    group2_name, group2_fields
                )
                
                if conflict_analysis['conflict_strength'] > self.conflicting_signals_threshold:
                    conflict_key = f"{group1_name}_vs_{group2_name}"
                    self.conflict_patterns[conflict_key] = conflict_analysis
        
        print(f"   Найдено {len(self.conflict_patterns)} конфликтных паттернов")
        
        # Детали по топ конфликтам
        sorted_conflicts = sorted(
            self.conflict_patterns.items(),
            key=lambda x: x[1]['conflict_strength'],
            reverse=True
        )
        
        for conflict_name, analysis in sorted_conflicts[:3]:
            print(f"     {conflict_name}: конфликт {analysis['conflict_strength']:.3f}")
    
    def _analyze_group_conflicts(self, group1_name, group1_fields, group2_name, group2_fields):
        """Анализ конфликтов между двумя группами полей"""
        
        # Получаем активность групп
        group1_activity = self._calculate_group_activity(group1_fields)
        group2_activity = self._calculate_group_activity(group2_fields)
        
        # Ищем периоды одновременной активации
        both_active = (group1_activity > 0) & (group2_activity > 0)
        
        if both_active.sum() < 10:  # Недостаточно данных
            return {'conflict_strength': 0, 'conflict_events': 0}
        
        # Анализируем направления сигналов
        group1_direction = self._determine_signal_direction(group1_fields, group1_activity > 0)
        group2_direction = self._determine_signal_direction(group2_fields, group2_activity > 0)
        
        # Ищем противоположные сигналы
        conflicting_periods = both_active & (group1_direction * group2_direction < 0)
        
        # Анализируем результативность в конфликтные периоды
        events_during_conflicts = self.targets['is_event'][conflicting_periods]
        
        if len(events_during_conflicts) > 0:
            conflict_success_rate = events_during_conflicts.mean()
            baseline_success_rate = self.targets['is_event'].mean()
            
            # Сила конфликта: насколько снижается успешность
            conflict_strength = max(0, baseline_success_rate - conflict_success_rate)
            
            return {
                'conflict_strength': conflict_strength,
                'conflict_events': conflicting_periods.sum(),
                'conflict_success_rate': conflict_success_rate,
                'baseline_success_rate': baseline_success_rate,
                'group1_activity_rate': (group1_activity > 0).mean(),
                'group2_activity_rate': (group2_activity > 0).mean()
            }
        
        return {'conflict_strength': 0, 'conflict_events': 0}
    
    def _calculate_group_activity(self, group_fields):
        """Расчет активности группы полей"""
        group_activity = pd.Series(0.0, index=self.features.index)
        
        for field in group_fields:
            # Ищем все вариации поля в данных
            field_columns = [col for col in self.features.columns if col.startswith(field)]
            
            for col in field_columns:
                field_data = pd.to_numeric(self.features[col], errors='coerce').fillna(0)
                group_activity += np.abs(field_data)
        
        return group_activity
    
    def _determine_signal_direction(self, group_fields, active_mask):
        """Определение направления сигнала группы"""
        
        # Простая эвристика: положительные значения = +1, отрицательные = -1
        directions = pd.Series(0.0, index=self.features.index)
        
        for field in group_fields:
            field_columns = [col for col in self.features.columns if col.startswith(field)]
            
            for col in field_columns:
                field_data = pd.to_numeric(self.features[col], errors='coerce').fillna(0)
                directions += np.sign(field_data)
        
        # Возвращаем направление только в активные периоды
        result = pd.Series(0.0, index=self.features.index)
        result[active_mask] = np.sign(directions[active_mask])
        
        return result
    
    def _detect_false_activations(self):
        """Детекция ложных активаций"""
        print("🎭 Детекция ложных активаций...")
        
        self.false_signal_analysis = {}
        
        # Анализируем каждое поле на предмет ложных сигналов
        for feature in self.features.columns:
            false_analysis = self._analyze_false_signals_for_field(feature)
            
            if false_analysis['false_positive_rate'] > 0.7:  # Высокий уровень ложных сигналов
                self.false_signal_analysis[feature] = false_analysis
        
        print(f"   Найдено {len(self.false_signal_analysis)} полей с высоким уровнем ложных сигналов")
        
        # Топ полей с ложными сигналами
        sorted_false = sorted(
            self.false_signal_analysis.items(),
            key=lambda x: x[1]['false_positive_rate'],
            reverse=True
        )
        
        for field, analysis in sorted_false[:5]:
            print(f"     {field}: ложные сигналы {analysis['false_positive_rate']:.1%}")
    
    def _analyze_false_signals_for_field(self, field):
        """Анализ ложных сигналов для поля"""
        
        field_data = pd.to_numeric(self.features[field], errors='coerce').fillna(0)
        
        if field_data.std() < 0.01:
            return {'false_positive_rate': 0, 'activations': 0}
        
        # Определяем активации поля
        threshold = np.percentile(np.abs(field_data), 80)
        activations = np.abs(field_data) > threshold
        
        if activations.sum() < 5:
            return {'false_positive_rate': 0, 'activations': 0}
        
        # Анализируем результативность активаций
        events = self.targets['is_event']
        
        # Проверяем события в ближайшие периоды после активации
        false_positives = 0
        total_activations = 0
        
        for idx in np.where(activations)[0]:
            total_activations += 1
            
            # Проверяем события в следующие 5 периодов
            future_window = slice(idx + 1, min(len(events), idx + 6))
            future_events = events.iloc[future_window] if future_window.start < len(events) else []
            
            # Если нет событий в ближайшем будущем - ложный сигнал
            if len(future_events) == 0 or future_events.sum() == 0:
                false_positives += 1
        
        false_positive_rate = false_positives / max(1, total_activations)
        
        return {
            'false_positive_rate': false_positive_rate,
            'activations': total_activations,
            'false_positives': false_positives,
            'threshold': threshold
        }
    
    def _build_veto_rules(self):
        """Построение VETO правил"""
        print("📋 Построение VETO правил...")
        
        self.veto_rules = {
            'blocking_fields': {},
            'conflict_rules': {},
            'false_signal_filters': {},
            'combination_rules': {}
        }
        
        # Правила блокировки
        for field, analysis in self.blocking_fields.items():
            if analysis['blocking_strength'] > 0.15:  # Значимые блокираторы
                self.veto_rules['blocking_fields'][field] = {
                    'action': 'block_signal',
                    'threshold': analysis['threshold'],
                    'blocking_strength': analysis['blocking_strength'],
                    'condition': f"if abs({field}) > {analysis['threshold']:.3f} then BLOCK"
                }
        
        # Правила конфликтов
        for conflict_name, analysis in self.conflict_patterns.items():
            if analysis['conflict_strength'] > 0.2:  # Значимые конфликты
                self.veto_rules['conflict_rules'][conflict_name] = {
                    'action': 'reduce_confidence',
                    'conflict_strength': analysis['conflict_strength'],
                    'confidence_penalty': min(0.5, analysis['conflict_strength']),
                    'condition': f"if {conflict_name} conflicting then REDUCE_CONFIDENCE"
                }
        
        # Фильтры ложных сигналов
        for field, analysis in self.false_signal_analysis.items():
            if analysis['false_positive_rate'] > 0.8:  # Очень высокий уровень ложных сигналов
                self.veto_rules['false_signal_filters'][field] = {
                    'action': 'ignore_field',
                    'false_positive_rate': analysis['false_positive_rate'],
                    'condition': f"IGNORE {field} (false positive rate: {analysis['false_positive_rate']:.1%})"
                }
        
        # Комбинированные правила
        self._build_combination_rules()
        
        print(f"   Создано {len(self.veto_rules['blocking_fields'])} правил блокировки")
        print(f"   Создано {len(self.veto_rules['conflict_rules'])} правил конфликтов")
        print(f"   Создано {len(self.veto_rules['false_signal_filters'])} фильтров ложных сигналов")
    
    def _build_combination_rules(self):
        """Построение комбинированных правил"""
        
        # Правило минимального количества подтверждающих сигналов
        self.veto_rules['combination_rules']['min_confirmations'] = {
            'action': 'require_multiple_signals',
            'min_signals': self.min_confirming_signals,
            'condition': f"REQUIRE at least {self.min_confirming_signals} confirming signals"
        }
        
        # Правило высокой волатильности
        if 'volatility' in self.data.columns:
            volatility_data = pd.to_numeric(self.data['volatility'], errors='coerce').fillna(0)
            high_vol_threshold = volatility_data.quantile(0.9)
            
            self.veto_rules['combination_rules']['high_volatility'] = {
                'action': 'block_during_high_volatility',
                'threshold': high_vol_threshold,
                'condition': f"if volatility > {high_vol_threshold:.3f} then BLOCK"
            }
        
        # Правило низкой уверенности
        self.veto_rules['combination_rules']['low_confidence'] = {
            'action': 'block_low_confidence',
            'threshold': self.low_confidence_threshold,
            'condition': f"if confidence < {self.low_confidence_threshold:.2f} then BLOCK"
        }
    
    def _validate_veto_system(self):
        """Валидация VETO системы"""
        print("✅ Валидация VETO системы...")
        
        # Применяем VETO правила и сравниваем результаты
        original_signals = self._create_baseline_signals()
        filtered_signals = self._apply_veto_rules(original_signals)
        
        # Метрики до и после применения VETO
        original_accuracy = self._calculate_signal_accuracy(original_signals)
        filtered_accuracy = self._calculate_signal_accuracy(filtered_signals)
        
        # Количество заблокированных сигналов
        blocked_signals = (original_signals == 1) & (filtered_signals == 0)
        blocked_count = blocked_signals.sum()
        
        # Эффективность блокировки
        blocked_false_positives = blocked_signals & (self.targets['is_event'] == 0)
        blocked_true_positives = blocked_signals & (self.targets['is_event'] == 1)
        
        veto_effectiveness = blocked_false_positives.sum() / max(1, blocked_count)
        
        validation_results = {
            'original_accuracy': original_accuracy,
            'filtered_accuracy': filtered_accuracy,
            'accuracy_improvement': filtered_accuracy - original_accuracy,
            'blocked_signals': int(blocked_count),
            'blocked_false_positives': int(blocked_false_positives.sum()),
            'blocked_true_positives': int(blocked_true_positives.sum()),
            'veto_effectiveness': veto_effectiveness,
            'signal_reduction': blocked_count / max(1, original_signals.sum())
        }
        
        self.veto_rules['validation'] = validation_results
        
        print(f"   Исходная точность: {original_accuracy:.3f}")
        print(f"   Точность с VETO: {filtered_accuracy:.3f}")
        print(f"   Улучшение: {validation_results['accuracy_improvement']:.3f}")
        print(f"   Заблокировано сигналов: {blocked_count}")
        print(f"   Эффективность VETO: {veto_effectiveness:.1%}")
        
        return validation_results
    
    def _create_baseline_signals(self):
        """Создание базовых сигналов для валидации"""
        
        # Простая baseline модель на основе активности полей
        signal_strength = pd.Series(0.0, index=self.features.index)
        
        # Суммируем активность всех групп
        for group_name, group_fields in self.field_groups.items():
            group_activity = self._calculate_group_activity(group_fields)
            signal_strength += group_activity
        
        # Преобразуем в бинарные сигналы
        threshold = signal_strength.quantile(0.8)
        baseline_signals = (signal_strength > threshold).astype(int)
        
        return baseline_signals
    
    def _apply_veto_rules(self, signals):
        """Применение VETO правил к сигналам"""
        
        filtered_signals = signals.copy()
        
        # Применяем блокирующие поля
        for field, rule in self.veto_rules['blocking_fields'].items():
            if field in self.features.columns:
                field_data = pd.to_numeric(self.features[field], errors='coerce').fillna(0)
                blocking_mask = np.abs(field_data) > rule['threshold']
                
                # Блокируем сигналы при активации блокирующего поля
                filtered_signals[blocking_mask] = 0
        
        # Применяем фильтры ложных сигналов
        false_fields = list(self.veto_rules['false_signal_filters'].keys())
        
        # Снижаем вес сигналов от полей с высоким уровнем ложных срабатываний
        for field in false_fields:
            if field in self.features.columns:
                field_data = pd.to_numeric(self.features[field], errors='coerce').fillna(0)
                field_active = np.abs(field_data) > field_data.quantile(0.8)
                
                # Блокируем сигналы только от этого поля
                isolated_signals = field_active.astype(int)
                filtered_signals = filtered_signals & ~isolated_signals
        
        return filtered_signals
    
    def _calculate_signal_accuracy(self, signals):
        """Расчет точности сигналов"""
        if signals.sum() == 0:
            return 0
        
        events = self.targets['is_event']
        correct_signals = (signals == 1) & (events == 1)
        
        accuracy = correct_signals.sum() / signals.sum()
        return accuracy
    
    def apply_veto_to_scoring(self, scoring_features, scoring_weights):
        """
        Применение VETO правил к системе скоринга
        
        Args:
            scoring_features: список признаков для скоринга
            scoring_weights: словарь весов признаков
            
        Returns:
            filtered_features, adjusted_weights
        """
        print("🔧 Применение VETO правил к скорингу...")
        
        filtered_features = []
        adjusted_weights = {}
        
        # Фильтрация признаков
        for feature in scoring_features:
            base_feature = feature.replace('_activated', '')
            
            # Проверка на блокирующие поля
            if base_feature in self.veto_rules['blocking_fields']:
                print(f"   Исключен блокирующий признак: {feature}")
                continue
            
            # Проверка на ложные сигналы
            if base_feature in self.veto_rules['false_signal_filters']:
                print(f"   Исключен ложно-сигнальный признак: {feature}")
                continue
            
            # Признак прошел фильтрацию
            filtered_features.append(feature)
            
            # Корректировка весов для конфликтных признаков
            original_weight = scoring_weights.get(feature, 0)
            adjusted_weight = original_weight
            
            # Снижение веса для конфликтных групп
            for conflict_name, conflict_rule in self.veto_rules['conflict_rules'].items():
                if any(group in conflict_name for group in self.field_groups.keys() 
                      if any(field in base_feature for field in self.field_groups[group])):
                    penalty = conflict_rule.get('confidence_penalty', 0)
                    adjusted_weight *= (1 - penalty)
            
            adjusted_weights[feature] = adjusted_weight
        
        print(f"   Фильтрация: {len(scoring_features)} → {len(filtered_features)} признаков")
        
        return filtered_features, adjusted_weights
    
    def save_veto_analysis(self, output_dir="results/veto_system"):
        """Сохранение результатов VETO анализа"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение VETO правил
        with open(output_path / "veto_rules.json", 'w') as f:
            # Конвертация для JSON совместимости
            json_rules = {}
            for category, rules in self.veto_rules.items():
                json_rules[category] = {}
                for rule_name, rule_data in rules.items():
                    if isinstance(rule_data, dict):
                        json_rules[category][rule_name] = {
                            k: float(v) if isinstance(v, (np.int64, np.float64)) else v 
                            for k, v in rule_data.items()
                        }
                    else:
                        json_rules[category][rule_name] = rule_data
            
            json.dump(json_rules, f, indent=2)
        
        # Сохранение блокирующих полей
        if self.blocking_fields:
            blocking_df = pd.DataFrame(self.blocking_fields).T
            blocking_df.to_csv(output_path / "blocking_fields.csv")
        
        # Сохранение конфликтных паттернов
        if self.conflict_patterns:
            conflict_df = pd.DataFrame(self.conflict_patterns).T
            conflict_df.to_csv(output_path / "conflict_patterns.csv")
        
        # Создание отчета
        self._create_veto_report(output_path)
        
        # Создание визуализаций
        self._create_veto_visualizations(output_path)
        
        print(f"✅ Результаты VETO анализа сохранены в {output_path}")
    
    def _create_veto_report(self, output_path):
        """Создание отчета по VETO системе"""
        
        report_lines = [
            "СИСТЕМА СТОП-ПОЛЕЙ И VETO ЛОГИКИ - ОТЧЕТ",
            "=" * 50,
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ПАРАМЕТРЫ СИСТЕМЫ:",
            f"Порог высокой волатильности: {self.high_volatility_threshold}",
            f"Порог конфликтующих сигналов: {self.conflicting_signals_threshold}",
            f"Порог низкой уверенности: {self.low_confidence_threshold}",
            f"Минимум подтверждающих сигналов: {self.min_confirming_signals}",
            "",
            "РЕЗУЛЬТАТЫ АНАЛИЗА:",
        ]
        
        # Блокирующие поля
        report_lines.extend([
            f"Найдено блокирующих полей: {len(self.blocking_fields)}",
        ])
        
        if self.blocking_fields:
            report_lines.append("Топ блокираторы:")
            sorted_blockers = sorted(
                self.blocking_fields.items(),
                key=lambda x: x[1]['blocking_strength'],
                reverse=True
            )
            for field, analysis in sorted_blockers[:10]:
                report_lines.append(
                    f"  {field}: блокировка {analysis['blocking_strength']:.3f}, "
                    f"ложные сигналы {analysis['false_positive_rate']:.1%}"
                )
        
        # Конфликтные паттерны
        report_lines.extend([
            "",
            f"Найдено конфликтных паттернов: {len(self.conflict_patterns)}",
        ])
        
        if self.conflict_patterns:
            for conflict_name, analysis in self.conflict_patterns.items():
                report_lines.append(
                    f"  {conflict_name}: конфликт {analysis['conflict_strength']:.3f}"
                )
        
        # Ложные сигналы
        report_lines.extend([
            "",
            f"Поля с высоким уровнем ложных сигналов: {len(self.false_signal_analysis)}",
        ])
        
        if self.false_signal_analysis:
            sorted_false = sorted(
                self.false_signal_analysis.items(),
                key=lambda x: x[1]['false_positive_rate'],
                reverse=True
            )
            for field, analysis in sorted_false[:10]:
                report_lines.append(
                    f"  {field}: ложные сигналы {analysis['false_positive_rate']:.1%}"
                )
        
        # Валидация
        if 'validation' in self.veto_rules:
            val = self.veto_rules['validation']
            report_lines.extend([
                "",
                "РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:",
                f"Исходная точность: {val['original_accuracy']:.3f}",
                f"Точность с VETO: {val['filtered_accuracy']:.3f}",
                f"Улучшение: {val['accuracy_improvement']:.3f}",
                f"Заблокировано сигналов: {val['blocked_signals']}",
                f"Эффективность VETO: {val['veto_effectiveness']:.1%}",
            ])
        
        report_lines.extend([
            "",
            "ФАЙЛЫ РЕЗУЛЬТАТОВ:",
            "veto_rules.json - правила VETO системы",
            "blocking_fields.csv - анализ блокирующих полей",
            "conflict_patterns.csv - паттерны конфликтов",
            "veto_effectiveness.png - график эффективности",
            "",
            "=" * 50
        ])
        
        with open(output_path / "veto_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _create_veto_visualizations(self, output_path):
        """Создание визуализаций VETO системы"""
        
        plt.style.use('default')
        
        # График 1: Эффективность блокирующих полей
        if self.blocking_fields:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Сила блокировки
            fields = list(self.blocking_fields.keys())[:10]  # Топ 10
            blocking_strengths = [self.blocking_fields[f]['blocking_strength'] for f in fields]
            
            bars1 = ax1.barh(range(len(fields)), blocking_strengths, color='red', alpha=0.7)
            ax1.set_yticks(range(len(fields)))
            ax1.set_yticklabels(fields)
            ax1.set_xlabel('Сила блокировки')
            ax1.set_title('Топ блокирующие поля')
            ax1.grid(True, alpha=0.3)
            
            # Частота ложных сигналов
            false_rates = [self.blocking_fields[f]['false_positive_rate'] for f in fields]
            
            bars2 = ax2.barh(range(len(fields)), false_rates, color='orange', alpha=0.7)
            ax2.set_yticks(range(len(fields)))
            ax2.set_yticklabels(fields)
            ax2.set_xlabel('Частота ложных сигналов')
            ax2.set_title('Ложные сигналы блокирующих полей')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "blocking_fields_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # График 2: Эффективность VETO системы
        if 'validation' in self.veto_rules:
            val = self.veto_rules['validation']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['Исходная точность', 'Точность с VETO']
            values = [val['original_accuracy'], val['filtered_accuracy']]
            colors = ['lightblue', 'lightgreen']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8)
            
            # Добавление значений на столбцы
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('Точность')
            ax.set_title('Эффективность VETO системы')
            ax.set_ylim(0, max(values) * 1.1)
            ax.grid(True, alpha=0.3)
            
            # Добавление информации о заблокированных сигналах
            ax.text(0.5, max(values) * 0.5, 
                   f"Заблокировано сигналов: {val['blocked_signals']}\n"
                   f"Эффективность: {val['veto_effectiveness']:.1%}",
                   ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(output_path / "veto_effectiveness.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Функция для тестирования VETO системы"""
    
    print("🧪 Тестирование системы стоп-полей и VETO логики...")
    
    # Создание тестовых данных
    np.random.seed(42)
    n_samples = 500
    
    # Генерация синтетических признаков
    n_features = 50
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Добавление "блокирующих" полей (антикоррелированных с событиями)
    X['blocker_1'] = np.random.randn(n_samples)
    X['blocker_2'] = np.random.randn(n_samples)
    
    # Генерация событий
    signal_strength = X[['feature_0', 'feature_1', 'feature_2']].sum(axis=1)
    noise = np.random.randn(n_samples) * 0.5
    
    # События происходят при высоком сигнале, но блокируются блокираторами
    events = (signal_strength + noise > 1.5) & (X['blocker_1'] < 1) & (X['blocker_2'] < 1)
    
    # Создание данных и целей
    data = pd.DataFrame({'close': 50000 + np.cumsum(np.random.randn(n_samples))})
    targets = pd.DataFrame({'is_event': events.astype(int)})
    
    # Тестирование VETO системы
    veto_system = VetoSystem()
    veto_rules = veto_system.analyze_veto_patterns(data, X, targets)
    
    # Сохранение результатов
    veto_system.save_veto_analysis("results/test_veto_system")
    
    print("✅ Тестирование VETO системы завершено")


if __name__ == "__main__":
    main()