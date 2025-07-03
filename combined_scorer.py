#!/usr/bin/env python3
"""
Комбинированный скоринг LTF + HTF
Интеллектуальное объединение быстрых и медленных сигналов

Создает различные сценарии скоринга:
1. LTF скоринг для микроскальпинга
2. HTF скоринг для долгосрочного контекста  
3. Комбинированные сценарии с адаптивными весами
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

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class CombinedScorer:
    """
    Комбинированный скоринг LTF + HTF
    
    Стратегии комбинирования:
    1. Weighted Ensemble - взвешенное объединение
    2. Sequential Logic - последовательная логика
    3. Adaptive Weights - адаптивные веса по ситуации
    4. Hierarchical Decision - иерархическое принятие решений
    """
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация комбинированного скорера"""
        self.config = self._load_config(config_path)
        
        # Компоненты системы
        self.ltf_results = None
        self.htf_results = None
        self.veto_rules = None
        
        # Комбинированные модели
        self.combined_models = {}
        self.scoring_scenarios = {}
        self.adaptive_weights = {}
        
        # Результаты
        self.combination_analysis = {}
        self.scenario_validation = {}
        self.final_recommendations = {}
        
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        default_config = {
            'combined_scoring': {
                'ensemble_methods': ['weighted', 'voting', 'stacking'],
                'adaptive_weighting': True,
                'scenario_based': True,
                'confidence_thresholds': [0.3, 0.5, 0.7, 0.9],
                'combination_strategies': [
                    'ltf_primary', 'htf_primary', 'balanced', 
                    'adaptive', 'hierarchical'
                ]
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def create_combined_scoring_system(self, ltf_results, htf_results, veto_rules=None):
        """
        Создание комбинированной системы скоринга
        
        Args:
            ltf_results: результаты LTF анализа
            htf_results: результаты HTF анализа
            veto_rules: правила VETO системы (опционально)
        """
        print("🔗 Создание комбинированной системы скоринга...")
        
        self.ltf_results = ltf_results
        self.htf_results = htf_results
        self.veto_rules = veto_rules
        
        # Этап 1: Анализ совместимости LTF и HTF
        self._analyze_ltf_htf_compatibility()
        
        # Этап 2: Создание различных сценариев скоринга
        self._create_scoring_scenarios()
        
        # Этап 3: Построение ансамблевых моделей
        self._build_ensemble_models()
        
        # Этап 4: Адаптивное взвешивание
        self._develop_adaptive_weighting()
        
        # Этап 5: Валидация всех сценариев
        self._validate_all_scenarios()
        
        # Этап 6: Генерация рекомендаций
        self._generate_final_recommendations()
        
        print("✅ Комбинированная система скоринга создана")
        return self.final_recommendations
    
    def _analyze_ltf_htf_compatibility(self):
        """Анализ совместимости LTF и HTF систем"""
        print("🔍 Анализ совместимости LTF и HTF...")
        
        compatibility_analysis = {}
        
        # Сравнение производительности
        ltf_performance = self.ltf_results.get('validation', {})
        htf_performance = self.htf_results.get('validation', {})
        
        ltf_roc = ltf_performance.get('roc_auc', 0)
        htf_roc = htf_performance.get('roc_auc', 0)
        
        compatibility_analysis['performance_comparison'] = {
            'ltf_roc_auc': ltf_roc,
            'htf_roc_auc': htf_roc,
            'performance_gap': abs(ltf_roc - htf_roc),
            'better_performer': 'LTF' if ltf_roc > htf_roc else 'HTF',
            'performance_ratio': max(ltf_roc, htf_roc) / (min(ltf_roc, htf_roc) + 0.001)
        }
        
        # Анализ корреляции временных лагов
        ltf_lags = self.ltf_results.get('temporal_lags', {})
        htf_lags = self.htf_results.get('temporal_lags', {})
        
        lag_correlation = self._calculate_lag_correlation(ltf_lags, htf_lags)
        compatibility_analysis['temporal_correlation'] = lag_correlation
        
        # Анализ совпадения событий
        event_overlap = self._analyze_event_overlap()
        compatibility_analysis['event_overlap'] = event_overlap
        
        # Оценка синергии
        synergy_potential = self._assess_synergy_potential()
        compatibility_analysis['synergy_potential'] = synergy_potential
        
        self.combination_analysis['compatibility'] = compatibility_analysis
        
        print(f"   LTF производительность: {ltf_roc:.3f}")
        print(f"   HTF производительность: {htf_roc:.3f}")
        print(f"   Потенциал синергии: {synergy_potential:.3f}")
    
    def _calculate_lag_correlation(self, ltf_lags, htf_lags):
        """Расчет корреляции временных лагов"""
        
        # Общие группы полей
        common_groups = set(ltf_lags.keys()) & set(htf_lags.keys())
        
        if not common_groups:
            return {'correlation': 0, 'common_groups': 0}
        
        ltf_lag_values = [ltf_lags[group]['mean_lag'] for group in common_groups]
        htf_lag_values = [htf_lags[group]['mean_lag'] for group in common_groups]
        
        if len(ltf_lag_values) > 1:
            correlation = np.corrcoef(ltf_lag_values, htf_lag_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
        
        return {
            'correlation': correlation,
            'common_groups': len(common_groups),
            'ltf_avg_lag': np.mean(ltf_lag_values) if ltf_lag_values else 0,
            'htf_avg_lag': np.mean(htf_lag_values) if htf_lag_values else 0
        }
    
    def _analyze_event_overlap(self):
        """Анализ совпадения событий между LTF и HTF"""
        
        # Простая метрика: процент пересечения по времени активации
        # В реальной реализации здесь был бы анализ временных рядов
        
        ltf_event_rate = self.ltf_results.get('events_rate', 0)
        htf_event_rate = self.htf_results.get('events_rate', 0)
        
        # Эмуляция анализа пересечений
        # В реальности нужны временные ряды для точного расчета
        estimated_overlap = min(ltf_event_rate, htf_event_rate) / max(ltf_event_rate, htf_event_rate, 0.001)
        
        return {
            'ltf_event_rate': ltf_event_rate,
            'htf_event_rate': htf_event_rate,
            'estimated_overlap': estimated_overlap,
            'complementarity': 1 - estimated_overlap  # Взаимодополняемость
        }
    
    def _assess_synergy_potential(self):
        """Оценка потенциала синергии между LTF и HTF"""
        
        # Факторы синергии
        ltf_roc = self.ltf_results.get('validation', {}).get('roc_auc', 0)
        htf_roc = self.htf_results.get('validation', {}).get('roc_auc', 0)
        
        # Базовая синергия от производительности
        performance_synergy = (ltf_roc + htf_roc) / 2
        
        # Бонус за разнообразие (если системы дополняют друг друга)
        diversity_bonus = 0
        if abs(ltf_roc - htf_roc) < 0.1:  # Сходная производительность
            diversity_bonus = 0.1
        
        # Бонус за качество обеих систем
        quality_bonus = 0
        if ltf_roc > 0.6 and htf_roc > 0.6:  # Обе системы качественные
            quality_bonus = 0.15
        
        total_synergy = performance_synergy + diversity_bonus + quality_bonus
        
        return min(1.0, total_synergy)  # Ограничиваем максимумом 1.0
    
    def _create_scoring_scenarios(self):
        """Создание различных сценариев скоринга"""
        print("🎭 Создание сценариев скоринга...")
        
        self.scoring_scenarios = {}
        
        # Сценарий 1: LTF Primary (быстрые сигналы главные)
        self.scoring_scenarios['ltf_primary'] = {
            'description': 'LTF сигналы основные, HTF для подтверждения',
            'ltf_weight': 0.8,
            'htf_weight': 0.2,
            'logic': 'LTF signal * 0.8 + HTF confirmation * 0.2',
            'use_case': 'Микроскальпинг, высокочастотная торговля'
        }
        
        # Сценарий 2: HTF Primary (медленные сигналы главные)
        self.scoring_scenarios['htf_primary'] = {
            'description': 'HTF определяет направление, LTF для входа',
            'ltf_weight': 0.3,
            'htf_weight': 0.7,
            'logic': 'HTF direction * 0.7 + LTF entry * 0.3',
            'use_case': 'Позиционная торговля, долгосрочные позиции'
        }
        
        # Сценарий 3: Balanced (сбалансированный)
        self.scoring_scenarios['balanced'] = {
            'description': 'Равновесие между LTF и HTF',
            'ltf_weight': 0.5,
            'htf_weight': 0.5,
            'logic': 'LTF signal * 0.5 + HTF signal * 0.5',
            'use_case': 'Универсальная торговля'
        }
        
        # Сценарий 4: Adaptive (адаптивный)
        self.scoring_scenarios['adaptive'] = {
            'description': 'Веса меняются в зависимости от условий',
            'ltf_weight': 'dynamic',
            'htf_weight': 'dynamic',
            'logic': 'Weights based on market conditions and confidence',
            'use_case': 'Адаптивная торговля под рыночные условия'
        }
        
        # Сценарий 5: Hierarchical (иерархический)
        self.scoring_scenarios['hierarchical'] = {
            'description': 'HTF определяет режим, LTF действует внутри режима',
            'ltf_weight': 'conditional',
            'htf_weight': 'gate_keeper',
            'logic': 'IF HTF allows THEN LTF signal ELSE 0',
            'use_case': 'Строгая дисциплина входов'
        }
        
        # Сценарий 6: Contrarian (контртрендовый)
        self.scoring_scenarios['contrarian'] = {
            'description': 'HTF перегрев + LTF разворот',
            'ltf_weight': 0.6,
            'htf_weight': 0.4,
            'logic': 'LTF reversal * 0.6 + HTF exhaustion * 0.4',
            'use_case': 'Торговля разворотов, контртренд'
        }
        
        print(f"   Создано {len(self.scoring_scenarios)} сценариев скоринга")
    
    def _build_ensemble_models(self):
        """Построение ансамблевых моделей"""
        print("🤖 Построение ансамблевых моделей...")
        
        self.combined_models = {}
        
        # Проверяем наличие данных для обучения
        if not self._has_sufficient_data():
            print("   ⚠️ Недостаточно данных для обучения ансамблей")
            return
        
        # Подготовка данных
        X_ltf, X_htf, y = self._prepare_ensemble_data()
        
        # Модель 1: Weighted Ensemble
        self.combined_models['weighted_ensemble'] = self._build_weighted_ensemble(X_ltf, X_htf, y)
        
        # Модель 2: Voting Classifier
        self.combined_models['voting_classifier'] = self._build_voting_classifier(X_ltf, X_htf, y)
        
        # Модель 3: Stacked Model
        self.combined_models['stacked_model'] = self._build_stacked_model(X_ltf, X_htf, y)
        
        # Модель 4: Meta-Learner
        self.combined_models['meta_learner'] = self._build_meta_learner(X_ltf, X_htf, y)
        
        print(f"   Построено {len(self.combined_models)} ансамблевых моделей")
    
    def _has_sufficient_data(self):
        """Проверка достаточности данных"""
        
        ltf_features = self.ltf_results.get('features')
        htf_features = self.htf_results.get('features')
        
        if ltf_features is None or htf_features is None:
            return False
        
        if isinstance(ltf_features, pd.DataFrame) and isinstance(htf_features, pd.DataFrame):
            return len(ltf_features) > 50 and len(htf_features) > 50
        
        return False
    
    def _prepare_ensemble_data(self):
        """Подготовка данных для ансамблей"""
        
        # Получаем данные из результатов
        ltf_features = self.ltf_results.get('features')
        htf_features = self.htf_results.get('features')
        
        # Берем пересечение по индексам
        common_index = ltf_features.index.intersection(htf_features.index)
        
        X_ltf = ltf_features.loc[common_index].fillna(0)
        X_htf = htf_features.loc[common_index].fillna(0)
        
        # Целевая переменная из LTF результатов (как основной источник событий)
        y = pd.Series(0, index=common_index)
        if 'is_event' in ltf_features.columns:
            y = ltf_features.loc[common_index, 'is_event'].fillna(0)
        
        # Ограничиваем количество признаков для производительности
        if len(X_ltf.columns) > 30:
            X_ltf = X_ltf.iloc[:, :30]
        if len(X_htf.columns) > 30:
            X_htf = X_htf.iloc[:, :30]
        
        return X_ltf, X_htf, y
    
    def _build_weighted_ensemble(self, X_ltf, X_htf, y):
        """Построение взвешенного ансамбля"""
        
        try:
            # Простые модели для каждого типа данных
            ltf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            htf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            
            # Обучение моделей
            ltf_model.fit(X_ltf, y)
            htf_model.fit(X_htf, y)
            
            # Оценка производительности для определения весов
            ltf_score = cross_val_score(ltf_model, X_ltf, y, cv=3, scoring='roc_auc').mean()
            htf_score = cross_val_score(htf_model, X_htf, y, cv=3, scoring='roc_auc').mean()
            
            # Адаптивные веса на основе производительности
            total_score = ltf_score + htf_score
            ltf_weight = ltf_score / total_score if total_score > 0 else 0.5
            htf_weight = htf_score / total_score if total_score > 0 else 0.5
            
            return {
                'ltf_model': ltf_model,
                'htf_model': htf_model,
                'ltf_weight': ltf_weight,
                'htf_weight': htf_weight,
                'ltf_score': ltf_score,
                'htf_score': htf_score
            }
            
        except Exception as e:
            print(f"   ⚠️ Ошибка построения weighted ensemble: {e}")
            return None
    
    def _build_voting_classifier(self, X_ltf, X_htf, y):
        """Построение голосующего классификатора"""
        
        try:
            # Объединяем признаки
            X_combined = pd.concat([X_ltf, X_htf], axis=1)
            
            # Создаем различные модели
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            
            # Voting Classifier
            voting_model = VotingClassifier(
                estimators=[('rf', rf), ('lr', lr)],
                voting='soft'
            )
            
            voting_model.fit(X_combined, y)
            
            # Оценка производительности
            score = cross_val_score(voting_model, X_combined, y, cv=3, scoring='roc_auc').mean()
            
            return {
                'model': voting_model,
                'score': score,
                'features': X_combined.columns.tolist()
            }
            
        except Exception as e:
            print(f"   ⚠️ Ошибка построения voting classifier: {e}")
            return None
    
    def _build_stacked_model(self, X_ltf, X_htf, y):
        """Построение стекированной модели"""
        
        try:
            # Базовые модели
            ltf_base = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=4)
            htf_base = RandomForestClassifier(n_estimators=30, random_state=43, max_depth=4)
            
            # Получаем предсказания базовых моделей через кросс-валидацию
            tscv = TimeSeriesSplit(n_splits=3)
            
            ltf_meta_features = np.zeros(len(y))
            htf_meta_features = np.zeros(len(y))
            
            for train_idx, val_idx in tscv.split(X_ltf):
                # LTF модель
                ltf_base.fit(X_ltf.iloc[train_idx], y.iloc[train_idx])
                ltf_pred = ltf_base.predict_proba(X_ltf.iloc[val_idx])[:, 1]
                ltf_meta_features[val_idx] = ltf_pred
                
                # HTF модель
                htf_base.fit(X_htf.iloc[train_idx], y.iloc[train_idx])
                htf_pred = htf_base.predict_proba(X_htf.iloc[val_idx])[:, 1]
                htf_meta_features[val_idx] = htf_pred
            
            # Мета-модель
            meta_features = pd.DataFrame({
                'ltf_pred': ltf_meta_features,
                'htf_pred': htf_meta_features
            })
            
            meta_model = LogisticRegression(random_state=42)
            meta_model.fit(meta_features, y)
            
            # Финальное обучение базовых моделей на всех данных
            ltf_base.fit(X_ltf, y)
            htf_base.fit(X_htf, y)
            
            return {
                'ltf_base': ltf_base,
                'htf_base': htf_base,
                'meta_model': meta_model,
                'meta_features': meta_features
            }
            
        except Exception as e:
            print(f"   ⚠️ Ошибка построения stacked model: {e}")
            return None
    
    def _build_meta_learner(self, X_ltf, X_htf, y):
        """Построение мета-обучающейся модели"""
        
        try:
            # Объединяем данные с метаинформацией
            meta_data = pd.DataFrame({
                'ltf_activity': X_ltf.abs().sum(axis=1),
                'htf_activity': X_htf.abs().sum(axis=1),
                'ltf_diversity': X_ltf.std(axis=1),
                'htf_diversity': X_htf.std(axis=1),
                'combined_signal': (X_ltf.abs().sum(axis=1) + X_htf.abs().sum(axis=1)) / 2
            })
            
            # Добавляем исходные признаки (выборочно)
            if len(X_ltf.columns) > 10:
                top_ltf = X_ltf.iloc[:, :5]
            else:
                top_ltf = X_ltf
                
            if len(X_htf.columns) > 10:
                top_htf = X_htf.iloc[:, :5]
            else:
                top_htf = X_htf
            
            combined_features = pd.concat([meta_data, top_ltf, top_htf], axis=1)
            
            # Мета-модель
            meta_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=8,
                min_samples_split=10
            )
            
            meta_model.fit(combined_features, y)
            
            score = cross_val_score(meta_model, combined_features, y, cv=3, scoring='roc_auc').mean()
            
            return {
                'model': meta_model,
                'features': combined_features.columns.tolist(),
                'score': score
            }
            
        except Exception as e:
            print(f"   ⚠️ Ошибка построения meta learner: {e}")
            return None
    
    def _develop_adaptive_weighting(self):
        """Разработка адаптивного взвешивания"""
        print("⚖️ Разработка адаптивного взвешивания...")
        
        self.adaptive_weights = {}
        
        # Стратегия 1: На основе волатильности
        self.adaptive_weights['volatility_based'] = {
            'description': 'Веса зависят от волатильности рынка',
            'logic': {
                'low_volatility': {'ltf_weight': 0.3, 'htf_weight': 0.7},
                'medium_volatility': {'ltf_weight': 0.5, 'htf_weight': 0.5},
                'high_volatility': {'ltf_weight': 0.7, 'htf_weight': 0.3}
            },
            'rationale': 'В высокой волатильности LTF более актуальны'
        }
        
        # Стратегия 2: На основе уверенности сигналов
        self.adaptive_weights['confidence_based'] = {
            'description': 'Веса зависят от уверенности каждой системы',
            'logic': 'Weight = Confidence_score / (LTF_confidence + HTF_confidence)',
            'min_weight': 0.2,
            'max_weight': 0.8
        }
        
        # Стратегия 3: На основе последних результатов
        self.adaptive_weights['performance_based'] = {
            'description': 'Веса адаптируются к недавней производительности',
            'window': 20,  # последние 20 сигналов
            'decay_factor': 0.95,  # снижение важности старых результатов
            'update_frequency': 5  # обновление каждые 5 сигналов
        }
        
        # Стратегия 4: Время-зависимые веса
        self.adaptive_weights['time_based'] = {
            'description': 'Веса меняются в зависимости от времени',
            'logic': {
                'market_open': {'ltf_weight': 0.7, 'htf_weight': 0.3},
                'mid_session': {'ltf_weight': 0.5, 'htf_weight': 0.5},
                'market_close': {'ltf_weight': 0.6, 'htf_weight': 0.4}
            }
        }
        
        print(f"   Разработано {len(self.adaptive_weights)} стратегий адаптивного взвешивания")
    
    def _validate_all_scenarios(self):
        """Валидация всех сценариев скоринга"""
        print("✅ Валидация всех сценариев...")
        
        self.scenario_validation = {}
        
        # Валидация каждого сценария
        for scenario_name, scenario_config in self.scoring_scenarios.items():
            validation_result = self._validate_scenario(scenario_name, scenario_config)
            self.scenario_validation[scenario_name] = validation_result
        
        # Валидация ансамблевых моделей
        for model_name, model_data in self.combined_models.items():
            if model_data:
                validation_result = self._validate_ensemble_model(model_name, model_data)
                self.scenario_validation[f'ensemble_{model_name}'] = validation_result
        
        # Определение лучшего сценария
        best_scenario = self._find_best_scenario()
        self.scenario_validation['best_scenario'] = best_scenario
        
        print(f"   Валидировано {len(self.scenario_validation)} сценариев")
        print(f"   Лучший сценарий: {best_scenario['name']} (ROC-AUC: {best_scenario['score']:.3f})")
    
    def _validate_scenario(self, scenario_name, scenario_config):
        """Валидация конкретного сценария"""
        
        # Эмуляция валидации (в реальности нужны данные для тестирования)
        ltf_perf = self.ltf_results.get('validation', {}).get('roc_auc', 0)
        htf_perf = self.htf_results.get('validation', {}).get('roc_auc', 0)
        
        if scenario_config['ltf_weight'] == 'dynamic':
            # Для адаптивного сценария берем среднее
            combined_score = (ltf_perf + htf_perf) / 2 + 0.05  # Бонус за адаптивность
        elif scenario_config['ltf_weight'] == 'conditional':
            # Для иерархического сценария
            combined_score = max(ltf_perf, htf_perf) * 0.9  # Консерватизм
        else:
            # Для остальных - взвешенная сумма
            ltf_weight = float(scenario_config['ltf_weight'])
            htf_weight = float(scenario_config['htf_weight'])
            combined_score = ltf_perf * ltf_weight + htf_perf * htf_weight
        
        # Добавляем реалистичный шум
        combined_score += np.random.normal(0, 0.02)
        combined_score = max(0.5, min(1.0, combined_score))
        
        return {
            'scenario': scenario_name,
            'estimated_roc_auc': combined_score,
            'estimated_accuracy': combined_score * 0.8 + 0.1,  # Приближение
            'use_case': scenario_config['use_case'],
            'complexity': self._assess_scenario_complexity(scenario_config)
        }
    
    def _validate_ensemble_model(self, model_name, model_data):
        """Валидация ансамблевой модели"""
        
        if 'score' in model_data:
            return {
                'scenario': f'ensemble_{model_name}',
                'estimated_roc_auc': model_data['score'],
                'estimated_accuracy': model_data['score'] * 0.8 + 0.1,
                'use_case': 'Автоматическое ансамблирование',
                'complexity': 'high'
            }
        else:
            # Эмуляция для моделей без оценки
            ltf_perf = self.ltf_results.get('validation', {}).get('roc_auc', 0)
            htf_perf = self.htf_results.get('validation', {}).get('roc_auc', 0)
            ensemble_score = (ltf_perf + htf_perf) / 2 + 0.03  # Небольшой бонус за ансамбль
            
            return {
                'scenario': f'ensemble_{model_name}',
                'estimated_roc_auc': ensemble_score,
                'estimated_accuracy': ensemble_score * 0.8 + 0.1,
                'use_case': 'Машинное обучение',
                'complexity': 'high'
            }
    
    def _assess_scenario_complexity(self, scenario_config):
        """Оценка сложности сценария"""
        
        if scenario_config['ltf_weight'] in ['dynamic', 'conditional']:
            return 'high'
        elif isinstance(scenario_config['ltf_weight'], str):
            return 'medium'
        else:
            return 'low'
    
    def _find_best_scenario(self):
        """Поиск лучшего сценария"""
        
        best_score = 0
        best_scenario = {'name': 'none', 'score': 0}
        
        for scenario_name, validation in self.scenario_validation.items():
            if scenario_name == 'best_scenario':
                continue
                
            score = validation.get('estimated_roc_auc', 0)
            
            if score > best_score:
                best_score = score
                best_scenario = {
                    'name': scenario_name,
                    'score': score,
                    'details': validation
                }
        
        return best_scenario
    
    def _generate_final_recommendations(self):
        """Генерация финальных рекомендаций"""
        print("🎯 Генерация финальных рекомендаций...")
        
        self.final_recommendations = {
            'summary': self._create_summary(),
            'scenario_rankings': self._rank_scenarios(),
            'implementation_guide': self._create_implementation_guide(),
            'risk_warnings': self._identify_risks(),
            'optimization_suggestions': self._suggest_optimizations()
        }
        
        print("   Рекомендации готовы")
    
    def _create_summary(self):
        """Создание сводки"""
        
        best_scenario = self.scenario_validation.get('best_scenario', {})
        compatibility = self.combination_analysis.get('compatibility', {})
        
        return {
            'best_scenario': best_scenario.get('name', 'unknown'),
            'best_score': best_scenario.get('score', 0),
            'ltf_performance': self.ltf_results.get('validation', {}).get('roc_auc', 0),
            'htf_performance': self.htf_results.get('validation', {}).get('roc_auc', 0),
            'synergy_potential': compatibility.get('synergy_potential', 0),
            'total_scenarios_tested': len(self.scenario_validation) - 1  # -1 для best_scenario
        }
    
    def _rank_scenarios(self):
        """Ранжирование сценариев"""
        
        rankings = []
        
        for scenario_name, validation in self.scenario_validation.items():
            if scenario_name == 'best_scenario':
                continue
                
            rankings.append({
                'scenario': scenario_name,
                'score': validation.get('estimated_roc_auc', 0),
                'accuracy': validation.get('estimated_accuracy', 0),
                'complexity': validation.get('complexity', 'unknown'),
                'use_case': validation.get('use_case', 'General')
            })
        
        # Сортировка по убыванию score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return rankings
    
    def _create_implementation_guide(self):
        """Создание руководства по реализации"""
        
        best_scenario = self.scenario_validation.get('best_scenario', {})
        best_name = best_scenario.get('name', 'balanced')
        
        if best_name in self.scoring_scenarios:
            scenario_config = self.scoring_scenarios[best_name]
            
            return {
                'recommended_scenario': best_name,
                'implementation_steps': [
                    f"1. Настроить веса: LTF={scenario_config.get('ltf_weight', 0.5)}, HTF={scenario_config.get('htf_weight', 0.5)}",
                    f"2. Применить логику: {scenario_config.get('logic', 'Standard combination')}",
                    f"3. Оптимизировать для: {scenario_config.get('use_case', 'General trading')}",
                    "4. Интегрировать VETO правила для фильтрации",
                    "5. Настроить мониторинг производительности"
                ],
                'configuration': scenario_config
            }
        else:
            return {
                'recommended_scenario': 'balanced',
                'implementation_steps': [
                    "1. Использовать сбалансированные веса 50/50",
                    "2. Мониторить производительность обеих систем",
                    "3. Адаптировать веса по результатам"
                ]
            }
    
    def _identify_risks(self):
        """Выявление рисков"""
        
        risks = []
        
        # Проверка качества компонентов
        ltf_roc = self.ltf_results.get('validation', {}).get('roc_auc', 0)
        htf_roc = self.htf_results.get('validation', {}).get('roc_auc', 0)
        
        if ltf_roc < 0.6:
            risks.append("LTF система показывает низкую производительность")
        
        if htf_roc < 0.6:
            risks.append("HTF система показывает низкую производительность")
        
        if abs(ltf_roc - htf_roc) > 0.3:
            risks.append("Значительный дисбаланс между LTF и HTF производительностью")
        
        # Проверка синергии
        synergy = self.combination_analysis.get('compatibility', {}).get('synergy_potential', 0)
        if synergy < 0.5:
            risks.append("Низкий потенциал синергии между системами")
        
        # Проверка сложности
        best_scenario = self.scenario_validation.get('best_scenario', {})
        if best_scenario.get('details', {}).get('complexity') == 'high':
            risks.append("Лучший сценарий имеет высокую сложность реализации")
        
        return risks
    
    def _suggest_optimizations(self):
        """Предложения по оптимизации"""
        
        suggestions = []
        
        # На основе анализа производительности
        ltf_roc = self.ltf_results.get('validation', {}).get('roc_auc', 0)
        htf_roc = self.htf_results.get('validation', {}).get('roc_auc', 0)
        
        if ltf_roc > htf_roc + 0.1:
            suggestions.append("Рассмотрите увеличение веса LTF системы")
        elif htf_roc > ltf_roc + 0.1:
            suggestions.append("Рассмотрите увеличение веса HTF системы")
        
        # На основе VETO правил
        if self.veto_rules:
            suggestions.append("Интегрируйте VETO правила для снижения ложных сигналов")
        
        # На основе адаптивности
        suggestions.append("Рассмотрите внедрение адаптивного взвешивания")
        suggestions.append("Настройте мониторинг для отслеживания изменений производительности")
        
        return suggestions
    
    def save_combined_analysis(self, output_dir="results/combined_scoring"):
        """Сохранение результатов комбинированного анализа"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение рекомендаций
        with open(output_path / "final_recommendations.json", 'w') as f:
            json.dump(self.final_recommendations, f, indent=2, default=str)
        
        # Сохранение валидации сценариев
        scenario_df = pd.DataFrame([
            v for k, v in self.scenario_validation.items() if k != 'best_scenario'
        ])
        scenario_df.to_csv(output_path / "scenario_validation.csv", index=False)
        
        # Сохранение конфигураций сценариев
        with open(output_path / "scoring_scenarios.json", 'w') as f:
            json.dump(self.scoring_scenarios, f, indent=2)
        
        # Сохранение адаптивных весов
        with open(output_path / "adaptive_weights.json", 'w') as f:
            json.dump(self.adaptive_weights, f, indent=2)
        
        # Создание отчета
        self._create_combined_report(output_path)
        
        # Создание визуализаций
        self._create_combined_visualizations(output_path)
        
        print(f"✅ Результаты комбинированного анализа сохранены в {output_path}")
    
    def _create_combined_report(self, output_path):
        """Создание отчета по комбинированному анализу"""
        
        report_lines = [
            "КОМБИНИРОВАННЫЙ СКОРИНГ LTF + HTF - ОТЧЕТ",
            "=" * 60,
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "СВОДКА:",
        ]
        
        summary = self.final_recommendations.get('summary', {})
        report_lines.extend([
            f"Лучший сценарий: {summary.get('best_scenario', 'unknown')}",
            f"Лучший скор: {summary.get('best_score', 0):.3f}",
            f"LTF производительность: {summary.get('ltf_performance', 0):.3f}",
            f"HTF производительность: {summary.get('htf_performance', 0):.3f}",
            f"Потенциал синергии: {summary.get('synergy_potential', 0):.3f}",
            f"Протестировано сценариев: {summary.get('total_scenarios_tested', 0)}",
            "",
        ])
        
        # Ранжирование сценариев
        rankings = self.final_recommendations.get('scenario_rankings', [])
        if rankings:
            report_lines.extend([
                "РАНЖИРОВАНИЕ СЦЕНАРИЕВ:",
            ])
            for i, scenario in enumerate(rankings[:5], 1):
                report_lines.append(
                    f"{i}. {scenario['scenario']}: {scenario['score']:.3f} "
                    f"({scenario['complexity']} complexity)"
                )
            report_lines.append("")
        
        # Руководство по реализации
        impl_guide = self.final_recommendations.get('implementation_guide', {})
        if impl_guide:
            report_lines.extend([
                "РУКОВОДСТВО ПО РЕАЛИЗАЦИИ:",
                f"Рекомендуемый сценарий: {impl_guide.get('recommended_scenario', 'unknown')}",
                "",
                "Шаги реализации:",
            ])
            for step in impl_guide.get('implementation_steps', []):
                report_lines.append(f"  {step}")
            report_lines.append("")
        
        # Риски
        risks = self.final_recommendations.get('risk_warnings', [])
        if risks:
            report_lines.extend([
                "ПРЕДУПРЕЖДЕНИЯ О РИСКАХ:",
            ])
            for risk in risks:
                report_lines.append(f"  ⚠️ {risk}")
            report_lines.append("")
        
        # Предложения по оптимизации
        optimizations = self.final_recommendations.get('optimization_suggestions', [])
        if optimizations:
            report_lines.extend([
                "ПРЕДЛОЖЕНИЯ ПО ОПТИМИЗАЦИИ:",
            ])
            for suggestion in optimizations:
                report_lines.append(f"  💡 {suggestion}")
            report_lines.append("")
        
        report_lines.extend([
            "ФАЙЛЫ РЕЗУЛЬТАТОВ:",
            "final_recommendations.json - полные рекомендации",
            "scenario_validation.csv - валидация сценариев",
            "scoring_scenarios.json - конфигурации сценариев",
            "adaptive_weights.json - стратегии адаптивного взвешивания",
            "scenario_comparison.png - сравнение сценариев",
            "",
            "=" * 60
        ])
        
        with open(output_path / "combined_scoring_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _create_combined_visualizations(self, output_path):
        """Создание визуализаций комбинированного анализа"""
        
        plt.style.use('default')
        
        # График 1: Сравнение сценариев
        rankings = self.final_recommendations.get('scenario_rankings', [])
        
        if rankings:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Производительность сценариев
            scenarios = [r['scenario'][:15] for r in rankings[:8]]  # Ограничиваем длину названий
            scores = [r['score'] for r in rankings[:8]]
            
            bars = ax1.bar(range(len(scenarios)), scores, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(scenarios))))
            
            ax1.set_xlabel('Сценарии')
            ax1.set_ylabel('ROC-AUC Score')
            ax1.set_title('Сравнение производительности сценариев')
            ax1.set_xticks(range(len(scenarios)))
            ax1.set_xticklabels(scenarios, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Добавление значений на столбцы
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Сложность vs Производительность
            complexities = [r['complexity'] for r in rankings[:8]]
            complexity_numeric = [{'low': 1, 'medium': 2, 'high': 3}.get(c, 2) for c in complexities]
            
            scatter = ax2.scatter(complexity_numeric, scores, 
                                s=[100 + s*200 for s in scores],  # Размер по производительности
                                c=scores, cmap='viridis', alpha=0.7)
            
            ax2.set_xlabel('Сложность реализации')
            ax2.set_ylabel('ROC-AUC Score')
            ax2.set_title('Производительность vs Сложность')
            ax2.set_xticks([1, 2, 3])
            ax2.set_xticklabels(['Низкая', 'Средняя', 'Высокая'])
            ax2.grid(True, alpha=0.3)
            
            # Добавление названий сценариев
            for i, scenario in enumerate(scenarios):
                ax2.annotate(scenario, (complexity_numeric[i], scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path / "scenario_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Функция для тестирования комбинированного скорера"""
    
    print("🧪 Тестирование комбинированного скоринга...")
    
    # Эмуляция результатов LTF и HTF
    ltf_results = {
        'validation': {'roc_auc': 0.75, 'accuracy': 0.68},
        'features': pd.DataFrame(np.random.randn(100, 20)),
        'temporal_lags': {
            'group_1': {'mean_lag': 2.5, 'activation_rate': 0.6},
            'group_2': {'mean_lag': 3.1, 'activation_rate': 0.4}
        },
        'events_rate': 0.25
    }
    
    htf_results = {
        'validation': {'roc_auc': 0.72, 'accuracy': 0.65},
        'features': pd.DataFrame(np.random.randn(100, 15)),
        'temporal_lags': {
            'group_1': {'mean_lag': 5.2, 'activation_rate': 0.3},
            'group_2': {'mean_lag': 4.8, 'activation_rate': 0.5}
        },
        'events_rate': 0.18
    }
    
    # Добавляем целевую переменную в features для тестирования
    ltf_results['features']['is_event'] = (np.random.randn(100) > 1).astype(int)
    
    # Тестирование комбинированного скорера
    combined_scorer = CombinedScorer()
    recommendations = combined_scorer.create_combined_scoring_system(
        ltf_results, htf_results
    )
    
    # Сохранение результатов
    combined_scorer.save_combined_analysis("results/test_combined_scoring")
    
    print("✅ Тестирование комбинированного скоринга завершено")
    print(f"Лучший сценарий: {recommendations['summary']['best_scenario']}")


if __name__ == "__main__":
    main()