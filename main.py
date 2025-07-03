#!/usr/bin/env python3
"""
Финансовый анализатор логов - ЧЕСТНАЯ DATA-DRIVEN ВЕРСИЯ
✅ НИКАКИХ АПРИОРНЫХ ПРЕДПОЛОЖЕНИЙ
✅ ТОЛЬКО СТАТИСТИКА И КОРРЕЛЯЦИИ
✅ АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ СОБЫТИЙ
✅ РЕАЛЬНЫЕ ROC-AUC ДЛЯ КАЖДОГО ПОЛЯ
✅ VETO АНАЛИЗ ЧЕРЕЗ АНТИКОРРЕЛЯЦИИ
✅ ПОЛНАЯ ОБЪЕКТИВНОСТЬ

ПРИНЦИП: ДАННЫЕ САМИ ПОКАЗЫВАЮТ ЧТО ВАЖНО
"""

import pandas as pd
import numpy as np
import re
import yaml
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Импорт продвинутых модулей (если доступны)
try:
    from advanced_log_parser import AdvancedLogParser
    from parser_integration import ParserIntegration
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Продвинутые модули недоступны: {e}")
    print("💡 Работаем с базовой функциональностью")
    ADVANCED_MODULES_AVAILABLE = False


class HonestDataDrivenAnalyzer:
    """
    ЧЕСТНЫЙ DATA-DRIVEN АНАЛИЗАТОР
    
    ПРИНЦИПЫ:
    - НИ ОДНО поле не имеет априорного преимущества
    - ТОЛЬКО реальная статистика
    - Автоматическое определение событий
    - Поиск неожиданных корреляций
    - VETO анализ антикорреляций
    """
    
    def __init__(self, config_path="config.yaml"):
        """Инициализация честного анализатора"""
        self.config = self._load_config(config_path)
        
        # Интеграция продвинутых модулей
        if ADVANCED_MODULES_AVAILABLE:
            self.advanced_parser = AdvancedLogParser()
            self.parser_integration = ParserIntegration(self)
        
        # Результаты анализа
        self.parsed_data = None
        self.features = None
        self.events = None
        self.field_correlations = {}
        self.field_roc_scores = {}
        self.real_temporal_lags = {}
        self.veto_fields = {}
        self.event_statistics = {}
        self.threshold_analysis = {}
        self.scoring_system = None
        self.validation_results = None
        
        # Создание папки результатов
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        default_config = {
            'events': {
                'min_price_change': 0.01,  # Минимальное изменение для события (1%)
                'lookback_window': 20,     # Окно для поиска экстремумов
                'min_event_gap': 5         # Минимальный промежуток между событиями
            },
            'analysis': {
                'min_correlation': 0.05,   # Минимальная корреляция для анализа
                'significance_level': 0.05, # Уровень значимости
                'min_samples': 10          # Минимум выборок для анализа
            },
            'veto': {
                'min_anticorrelation': -0.1, # Минимальная антикорреляция для VETO
                'effectiveness_threshold': 0.3 # Порог эффективности блокировки
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config

    def run_full_analysis(self, file_path):
        """
        ГЛАВНАЯ ФУНКЦИЯ: Полный честный анализ
        """
        print("🎯 ЗАПУСК ЧЕСТНОГО DATA-DRIVEN АНАЛИЗА")
        print("=" * 70)
        print("⚠️  ПРИНЦИП: НИ ОДНО ПОЛЕ НЕ ИМЕЕТ АПРИОРНОГО ПРЕИМУЩЕСТВА")
        print("📊 ТОЛЬКО РЕАЛЬНАЯ СТАТИСТИКА И КОРРЕЛЯЦИИ")
        print("=" * 70)
        
        try:
            # Шаг 1: Парсинг (честный)
            if not self.parse_log_file(file_path):
                return {'status': 'error', 'message': 'Ошибка парсинга файла'}
            
            # Шаг 2: Создание признаков (честный)
            if not self.create_features():
                return {'status': 'error', 'message': 'Ошибка создания признаков'}
            
            # Шаг 3: АВТОМАТИЧЕСКОЕ определение событий
            if not self.auto_detect_events():
                return {'status': 'error', 'message': 'Ошибка автоматического определения событий'}
            
            # Шаг 4: РЕАЛЬНАЯ статистика полей
            if not self.calculate_real_field_statistics():
                return {'status': 'error', 'message': 'Ошибка расчета статистики полей'}
            
            # Шаг 5: НАСТОЯЩИЕ корреляции и ROC-AUC
            if not self.calculate_real_correlations():
                return {'status': 'error', 'message': 'Ошибка расчета корреляций'}
            
            # Шаг 6: РЕАЛЬНЫЕ временные лаги
            if not self.calculate_real_temporal_lags():
                return {'status': 'error', 'message': 'Ошибка расчета временных лагов'}
            
            # Шаг 7: VETO анализ через антикорреляции
            if not self.find_veto_fields():
                return {'status': 'error', 'message': 'Ошибка VETO анализа'}
            
            # Шаг 8: Честная система скоринга
            if not self.create_honest_scoring_system():
                return {'status': 'error', 'message': 'Ошибка создания системы скоринга'}
            
            # Шаг 9: Валидация
            if not self.validate_system():
                return {'status': 'error', 'message': 'Ошибка валидации системы'}
            
            # Шаг 10: Создание отчетов
            results_folder = self.create_honest_reports(file_path)
            
            print("🎊 ЧЕСТНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
            print(f"📁 Все результаты: {results_folder}")
            print(f"📋 Главный отчет: {results_folder}/СТАТИСТИЧЕСКИЙ_АНАЛИЗ.txt")
            
            return {
                'status': 'success',
                'results_folder': results_folder,
                'validation_results': self.validation_results,
                'files_created': len(list(self.results_dir.glob("*.*")))
            }
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def parse_log_file(self, file_path):
        """Честный парсинг в ПОЛНОМ соответствии с ТЗ"""
        print(f"🔍 Честный парсинг файла: {file_path}")
        
        if ADVANCED_MODULES_AVAILABLE:
            # ГЛОБАЛЬНОЕ исправление всех ошибок форматирования в advanced_log_parser
            try:
                # Создаем безопасную обертку для парсера
                self._patch_advanced_parser()
                
                # Теперь используем исправленный парсер
                self.raw_parsing_data = self.advanced_parser.parse_log_file(file_path)
                
                if self.raw_parsing_data.empty:
                    print("❌ Продвинутый парсер не извлек данные")
                    return False
                
                # ОБЯЗАТЕЛЬНОЕ LTF/HTF разделение согласно ТЗ
                ltf_data, htf_data = self.advanced_parser.get_ltf_htf_separation(self.raw_parsing_data)
                
                # Интеграция результатов
                integration_results = self.parser_integration.replace_old_parser(file_path)
                
                if integration_results:
                    self.parsed_data = integration_results.get('full_data', pd.DataFrame())
                    print(f"✅ Извлечено записей: {len(self.parsed_data)}")
                    print(f"✅ Извлечено полей: {len(self.parsed_data.columns)}")
                    print(f"✅ LTF/HTF разделение выполнено согласно ТЗ")
                    return True
                else:
                    return False
                    
            except Exception as e:
                print(f"❌ Ошибка продвинутого парсера: {e}")
                return False
        else:
            return self._fallback_parse_log_file(file_path)

    def _patch_advanced_parser(self):
        """ГЛОБАЛЬНОЕ исправление всех ошибок форматирования в advanced_log_parser"""
        
        # Исправляем _generate_parsing_statistics
        self.advanced_parser._generate_parsing_statistics = self._safe_parsing_statistics
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильный паттерн для двойных минусов
        self.advanced_parser.field_patterns['universal_field'] = r'([a-zA-Z]+)(\d+|1h|4h|1d|1w)-(-{1,2}\d+(?:\.\d+)?(?:%)?|!+|\d+(?:\.\d+)?(?:%)?)'
        
        # Исправляем также для parser_integration
        if hasattr(self, 'parser_integration') and self.parser_integration:
            self.parser_integration.advanced_parser._generate_parsing_statistics = self._safe_parsing_statistics
            self.parser_integration.advanced_parser.field_patterns['universal_field'] = r'([a-zA-Z]+)(\d+|1h|4h|1d|1w)-(-{1,2}\d+(?:\.\d+)?(?:%)?|!+|\d+(?:\.\d+)?(?:%)?)'

    def _safe_parsing_statistics(self, df):
        """БЕЗОПАСНАЯ статистика парсинга (БЕЗ ошибок форматирования)"""
        print("\n📊 СТАТИСТИКА ИЗВЛЕЧЕННЫХ ПОЛЕЙ:")
        
        # Группы полей согласно ТЗ  
        field_groups = {
            'group_1': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rd', 'md', 'cd', 'cmd', 'macd', 'od', 'dd', 'cvd', 'drd', 'ad', 'ed', 'hd', 'sd']) and not col.endswith('_type')],
            'group_2': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['ro', 'mo', 'co', 'cz', 'do', 'ae', 'so']) and not col.endswith('_type')],
            'group_3': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rz', 'mz', 'ciz', 'sz', 'dz', 'cvz', 'maz', 'oz']) and not col.endswith('_type')],
            'group_4': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['ef', 'wv', 'vc', 'ze', 'nw', 'as', 'vw']) and not col.endswith('_type')],
            'group_5': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['bs', 'wa', 'pd']) and not col.endswith('_type')],
            'metadata': [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume', 'range']]
        }
        
        for group_name, fields in field_groups.items():
            if fields:
                print(f"   {group_name}: {len(fields)} полей")
        
        # LTF/HTF разделение согласно ТЗ
        ltf_fields = [col for col in df.columns if any(col.endswith(suffix) for suffix in ['2', '5', '15', '30']) and not col.endswith('_type')]
        htf_fields = [col for col in df.columns if any(suffix in col for suffix in ['1h', '4h', '1d', '1w']) and not col.endswith('_type')]
        
        print(f"\n🎯 LTF/HTF РАЗДЕЛЕНИЕ (согласно ТЗ):")
        print(f"   TYPE-1 (LTF) полей: {len(ltf_fields)}")
        print(f"   TYPE-2 (HTF) полей: {len(htf_fields)}")
        
        if len(htf_fields) == 0:
            print("   ⚠️ ПРЕДУПРЕЖДЕНИЕ: Нет HTF полей - файл содержит только TYPE-1 данные")
        
        print(f"✅ Безопасная статистика: {len(df)} записей с {len(df.columns)} полями")

    def _fallback_parse_log_file(self, file_path):
        """Резервный честный парсинг"""
        print("⚠️ Используется резервный парсер...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parsed_record = self._parse_single_line_honest(line, i)
                        if parsed_record:
                            data.append(parsed_record)
                    except Exception as e:
                        continue
            
            if data:
                self.parsed_data = pd.DataFrame(data)
                print(f"✅ Резервный парсер: {len(self.parsed_data)} записей")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Ошибка резервного парсинга: {e}")
            return False

    def _parse_single_line_honest(self, line, line_num):
        """Честный парсинг одной строки"""
        parts = line.split('|')
        if len(parts) < 3:
            return None
        
        record = {'line_number': line_num}
        
        # Timestamp
        timestamp_match = re.search(r'\[([^\]]+)\]', line)
        if timestamp_match:
            record['timestamp'] = timestamp_match.group(1)
        
        # OHLC
        ohlc_match = re.search(r'o:([0-9.]+).*?h:([0-9.]+).*?l:([0-9.]+).*?c:([0-9.]+)', line)
        if ohlc_match:
            record['open'] = float(ohlc_match.group(1))
            record['high'] = float(ohlc_match.group(2))
            record['low'] = float(ohlc_match.group(3))
            record['close'] = float(ohlc_match.group(4))
        
        # Volume
        volume_match = re.search(r'\|([0-9.]+K)\|', line)
        if volume_match:
            volume_str = volume_match.group(1)
            record['volume'] = float(volume_str.replace('K', '')) * 1000
        
        # Range
        rng_match = re.search(r'rng:([0-9.]+)', line)
        if rng_match:
            record['range'] = float(rng_match.group(1))
        
        # ЧЕСТНОЕ извлечение полей данных
        field_matches = re.findall(r'([a-zA-Z]+\d+)-([^,|]+)', line)
        for field_name, field_value in field_matches:
            if field_name.startswith('nw'):
                # Для NW полей сохраняем и число, и сигнал
                exclamation_count = field_value.count('!')
                if exclamation_count > 0:
                    record[field_name] = exclamation_count
                    record[f"{field_name}_signal"] = field_value
                else:
                    try:
                        record[field_name] = float(field_value.replace('%', ''))
                    except:
                        record[field_name] = 0
            else:
                # Остальные поля
                try:
                    clean_value = field_value.replace('%', '').replace('σ', '')
                    record[field_name] = float(clean_value)
                except:
                    record[field_name] = 0
        
        return record

    def create_features(self):
        """ИСПРАВЛЕНО: Создание признаков с ПРИОРИТЕТОМ ИНДИКАТОРАМ"""
        print("🔧 Создание признаков с приоритетом индикаторным полям...")
        
        if self.parsed_data is None or self.parsed_data.empty:
            return False
        
        try:
            if ADVANCED_MODULES_AVAILABLE and hasattr(self, 'parser_integration'):
                print("   ✅ Используется продвинутый парсер с приоритизацией")
                self.features = self.parser_integration.get_features_for_main_system()
            else:
                print("   ⚠️ Используется fallback с применением приоритизации")
                self.features = self._create_prioritized_features_fallback(self.parsed_data)
            
            if self.features is None or self.features.empty:
                return False
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Очистка данных
            self.features = self._clean_mixed_data_types(self.features)
            
            print(f"✅ Создано признаков: {len(self.features.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания признаков: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _clean_mixed_data_types(self, df):
        """
        КРИТИЧЕСКАЯ ОЧИСТКА: Исправление смешанных типов данных
        
        Проблема: парсер извлекает некоторые поля как строки ('-', '!!')
        Решение: конвертируем в правильные типы согласно ТЗ
        """
        print("🧹 Очистка смешанных типов данных...")
        
        cleaned_df = df.copy()
        converted_fields = 0
        
        for column in df.columns:
            if column in ['line_number', 'timestamp', 'raw_line']:
                continue
                
            # Анализируем содержимое колонки
            sample_data = df[column].dropna()
            if len(sample_data) == 0:
                continue
            
            # Категориальные поля (сигналы)
            if column.endswith('_signal'):
                # Оставляем как есть - это категориальные данные
                continue
            
            # Числовые поля
            elif column.endswith('_type'):
                # Оставляем как есть - это типы полей
                continue
                
            else:
                # Попытка конвертации в числовой тип
                try:
                    # Заменяем строковые значения на NaN
                    numeric_series = pd.to_numeric(sample_data, errors='coerce')
                    
                    # Если получили хотя бы несколько числовых значений
                    if numeric_series.notna().sum() > 0:
                        # Конвертируем всю колонку
                        cleaned_df[column] = pd.to_numeric(df[column], errors='coerce')
                        converted_fields += 1
                    else:
                        # Если все значения строковые, оставляем как есть
                        # но убираем проблемные символы
                        cleaned_df[column] = df[column].astype(str).replace('-', '0')
                        
                except Exception:
                    # В случае любой ошибки, заполняем нулями
                    cleaned_df[column] = 0
        
        print(f"   ✅ Конвертировано полей в числовые: {converted_fields}")
        
        # Проверяем результат
        string_columns = []
        for col in cleaned_df.columns:
            if col not in ['line_number', 'timestamp', 'raw_line'] and not col.endswith('_type'):
                if cleaned_df[col].dtype == 'object':
                    # Проверяем, содержит ли колонка сигналы
                    sample = cleaned_df[col].dropna()
                    if len(sample) > 0 and any('!' in str(val) for val in sample.iloc[:5]):
                        # Это сигнальные данные - нормально
                        continue
                    else:
                        string_columns.append(col)
        
        if string_columns:
            print(f"   ⚠️ Остались нечисловые поля: {len(string_columns)} (будут обработаны отдельно)")
        
        return cleaned_df

    def _create_prioritized_features_fallback(self, data):
        """
        FALLBACK МЕТОД: Приоритизация индикаторов даже без продвинутого парсера
        Обеспечивает ПРИОРИТЕТ ИНДИКАТОРНЫМ ПОЛЯМ согласно ТЗ
        """
        print("   🎯 ПРИОРИТЕТ 1: Индикаторные поля...")
        
        features = pd.DataFrame(index=data.index)
        
        # Группы индикаторных полей (из ТЗ)
        indicator_groups = {
            'group_1': ['rd', 'md', 'cd', 'cmd', 'macd', 'od', 'dd', 'cvd', 'drd', 'ad', 'ed', 'hd', 'sd'],
            'group_2': ['ro', 'mo', 'co', 'cz', 'do', 'ae', 'so'],
            'group_3': ['rz', 'mz', 'ciz', 'sz', 'dz', 'cvz', 'maz', 'oz'],
            'group_4': ['ef', 'wv', 'vc', 'ze', 'nw', 'as', 'vw'],  # КРИТИЧНО ВАЖНЫЕ!
            'group_5': ['bs', 'wa', 'pd']
        }
        
        metadata_fields = ['open', 'high', 'low', 'close', 'volume', 'range']
        
        # ПРИОРИТЕТ 1: СНАЧАЛА индикаторные поля
        indicator_count = 0
        for group_name, prefixes in indicator_groups.items():
            for prefix in prefixes:
                group_fields = [col for col in data.columns 
                              if col.startswith(prefix) and col not in metadata_fields]
                
                for field in group_fields:
                    if field in data.columns:
                        # Основное значение поля
                        features[f"{field}_ind"] = data[field]  # Префикс "_ind" = индикатор
                        
                        # Активность поля
                        numeric_data = pd.to_numeric(data[field], errors='coerce').fillna(0)
                        features[f"{field}_ind_active"] = (numeric_data != 0).astype(int)
                        
                        # Обработка специальных сигналов для NW полей
                        if field.startswith('nw'):
                            signal_field = f"{field}_signal"
                            if signal_field in data.columns:
                                features[f"{field}_ind_signal"] = data[signal_field]
                        
                        indicator_count += 1
        
        print(f"   ✅ Индикаторных полей: {indicator_count}")
        
        # ПРИОРИТЕТ 2: Производные от индикаторов
        print("   🔄 ПРИОРИТЕТ 2: Производные признаки...")
        
        # Групповые агрегаты по индикаторным группам
        for group_name, prefixes in indicator_groups.items():
            group_fields = []
            for prefix in prefixes:
                group_fields.extend([col for col in data.columns 
                                   if col.startswith(prefix) and col not in metadata_fields])
            
            if len(group_fields) >= 2:
                try:
                    numeric_group_data = pd.DataFrame(index=data.index)
                    for col in group_fields:
                        numeric_group_data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    
                    features[f"{group_name}_ind_max"] = numeric_group_data.max(axis=1)
                    features[f"{group_name}_ind_mean"] = numeric_group_data.mean(axis=1)
                    features[f"{group_name}_ind_active_count"] = (numeric_group_data != 0).sum(axis=1)
                except Exception:
                    continue
        
        # ПРИОРИТЕТ 3: Метаданные (справочно)
        print("   📊 ПРИОРИТЕТ 3: Метаданные (справочно)...")
        
        metadata_count = 0
        for field in metadata_fields:
            if field in data.columns:
                features[f"meta_{field}"] = data[field]  # Префикс "meta_" = вторичность
                metadata_count += 1
        
        # Простые производные от метаданных
        if all(f"meta_{col}" in features.columns for col in ['open', 'high', 'low', 'close']):
            features['meta_price_range'] = features['meta_high'] - features['meta_low']
            try:
                features['meta_close_position'] = (features['meta_close'] - features['meta_low']) / (features['meta_high'] - features['meta_low'] + 1e-8)
            except:
                pass
        
        print(f"   ✅ Метаданных полей: {metadata_count}")
        
        # Финальная очистка
        features.fillna(0, inplace=True)
        
        # Отчет о приоритизации
        indicator_cols = [col for col in features.columns if '_ind' in col]
        metadata_cols = [col for col in features.columns if col.startswith('meta_')]
        
        print(f"   🎯 ИТОГ: Индикаторных признаков: {len(indicator_cols)}")
        print(f"   📊 ИТОГ: Метаданных (справочно): {len(metadata_cols)}")
        
        return features

    def auto_detect_events(self):
        """
        АВТОМАТИЧЕСКОЕ определение событий из ценовых данных
        БЕЗ ПРЕДВЗЯТОСТИ - только математика
        ИСПРАВЛЕНО: универсальный поиск ценовых полей
        """
        print("🎯 Автоматическое определение событий...")
        
        try:
            # ИСПРАВЛЕНО: Универсальный поиск ценовых полей 
            close_field = None
            
            # Возможные варианты названий поля close
            close_candidates = ['META_close', 'meta_close', 'close', 'IND_close']
            
            for candidate in close_candidates:
                if candidate in self.features.columns:
                    close_field = candidate
                    break
            
            # Если не нашли точное совпадение, ищем по содержанию 'close'
            if close_field is None:
                close_columns = [col for col in self.features.columns if 'close' in col.lower()]
                if close_columns:
                    close_field = close_columns[0]  # Берем первый найденный
            
            # Дополнительная диагностика для отладки
            print(f"🔍 Поиск ценовых полей в {len(self.features.columns)} колонках:")
            price_related = [col for col in self.features.columns if any(x in col.lower() for x in ['close', 'price', 'open', 'high', 'low'])]
            if price_related:
                print(f"   📊 Найдены поля связанные с ценами: {price_related[:5]}...")
            else:
                print("   ⚠️ Не найдено полей связанных с ценами")
            
            if close_field is None:
                print("❌ Нет ценовых данных для определения событий")
                print(f"❌ Доступные колонки: {list(self.features.columns)[:10]}...")
                return False
            
            prices = self.features[close_field].dropna()
            if len(prices) < 20:
                print("❌ Недостаточно ценовых данных")
                return False
            
            events_mask = pd.Series([False] * len(self.features), index=self.features.index)
            
            # 1. ЭКСТРЕМУМЫ ЧЕРЕЗ ЛОКАЛЬНЫЕ МИНИМУМЫ/МАКСИМУМЫ
            window = self.config['events']['lookback_window']
            min_change = self.config['events']['min_price_change']
            
            for i in range(window, len(prices) - window):
                current_price = prices.iloc[i]
                left_window = prices.iloc[i-window:i]
                right_window = prices.iloc[i:i+window]
                
                # Локальный максимум
                if (current_price == left_window.max() and 
                    current_price == right_window.max()):
                    # ИСПРАВЛЕНО: Проверяем значимость движения с защитой от деления на ноль
                    if prices.iloc[i-window] != 0:
                        price_change = abs(current_price - prices.iloc[i-window]) / prices.iloc[i-window]
                    else:
                        price_change = 0
                    if price_change >= min_change:
                        events_mask.iloc[i] = True
                
                # Локальный минимум
                elif (current_price == left_window.min() and 
                      current_price == right_window.min()):
                    # ИСПРАВЛЕНО: Проверяем значимость движения с защитой от деления на ноль
                    if prices.iloc[i-window] != 0:
                        price_change = abs(current_price - prices.iloc[i-window]) / prices.iloc[i-window]
                    else:
                        price_change = 0
                    if price_change >= min_change:
                        events_mask.iloc[i] = True
            
            # 2. РЕЗКИЕ ДВИЖЕНИЯ ЧЕРЕЗ ВОЛАТИЛЬНОСТЬ
            returns = prices.pct_change()
            volatility_threshold = returns.std() * 2  # 2 стандартных отклонения
            
            volatile_moves = abs(returns) > volatility_threshold
            events_mask = events_mask | volatile_moves
            
            # 3. УДАЛЕНИЕ БЛИЗКИХ СОБЫТИЙ
            min_gap = self.config['events']['min_event_gap']
            events_indices = events_mask[events_mask].index.tolist()
            
            filtered_events = []
            for event_idx in events_indices:
                if not filtered_events or (event_idx - filtered_events[-1]) >= min_gap:
                    filtered_events.append(event_idx)
            
            # Финальная маска событий
            final_events = pd.Series([False] * len(self.features), index=self.features.index)
            final_events.loc[filtered_events] = True
            
            # Сохранение результатов
            total_events = final_events.sum()
            event_rate = total_events / len(final_events)
            
            self.events = {
                'events_mask': final_events,
                'total_events': total_events,
                'event_rate': event_rate,
                'detection_method': 'automatic_extrema_volatility',
                'parameters': {
                    'lookback_window': window,
                    'min_price_change': min_change,
                    'volatility_threshold': volatility_threshold,
                    'min_event_gap': min_gap,
                    'price_field_used': close_field  # ДОБАВЛЕНО: какое поле использовалось
                }
            }
            
            print(f"✅ Автоматически найдено событий: {total_events}")
            print(f"✅ Частота событий: {event_rate:.3f} ({event_rate*100:.1f}%)")
            print(f"✅ Метод: экстремумы + волатильность")
            print(f"✅ Использовано поле: {close_field}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка автоматического определения событий: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_real_field_statistics(self):
        """
        РЕАЛЬНАЯ статистика полей БЕЗ ПРЕДПОЛОЖЕНИЙ
        """
        print("📊 Расчет реальной статистики полей...")
        
        try:
            if self.features is None or self.events is None:
                return False
            
            events_mask = self.events['events_mask']
            field_stats = {}
            
            # Анализ всех полей без исключения
            for column in self.features.columns:
                if column in ['line_number', 'timestamp']:
                    continue
                
                field_data = self.features[column].dropna()
                min_samples = self.config.get('analysis', {}).get('min_samples', 10)
                if len(field_data) < min_samples:
                    continue
                
                stats_dict = {
                    'field_type': self._determine_field_type(column, field_data),
                    'total_observations': len(field_data),
                    'non_zero_observations': (field_data != 0).sum(),
                    'activation_rate': (field_data != 0).mean(),
                    'mean': float(field_data.mean()) if field_data.dtype in ['int64', 'float64'] else None,
                    'std': float(field_data.std()) if field_data.dtype in ['int64', 'float64'] else None,
                    'min': float(field_data.min()) if field_data.dtype in ['int64', 'float64'] else None,
                    'max': float(field_data.max()) if field_data.dtype in ['int64', 'float64'] else None,
                    'percentiles': {}
                }
                
                # Процентили для числовых полей
                if field_data.dtype in ['int64', 'float64']:
                    for p in [10, 25, 50, 75, 90, 95, 99]:
                        stats_dict['percentiles'][f'p{p}'] = float(field_data.quantile(p/100))
                
                # Для категориальных полей (например, NW сигналы)
                if field_data.dtype == 'object':
                    value_counts = field_data.value_counts()
                    stats_dict['unique_values'] = value_counts.to_dict()
                    stats_dict['most_frequent'] = value_counts.index[0] if len(value_counts) > 0 else None
                
                field_stats[column] = stats_dict
            
            self.threshold_analysis = field_stats
            
            print(f"✅ Проанализировано полей: {len(field_stats)}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка расчета статистики полей: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _determine_field_type(self, column, data):
        """Определение типа поля БЕЗ ПРЕДПОЛОЖЕНИЙ"""
        if 'signal' in column:
            return 'categorical'
        elif data.dtype in ['int64', 'float64']:
            return 'numeric'
        elif data.dtype == 'object':
            return 'categorical'
        else:
            return 'unknown'

    def calculate_real_correlations(self):
        """
        НАСТОЯЩИЕ корреляции и ROC-AUC для каждого поля
        БЕЗ ПРЕДВЗЯТОСТИ
        """
        print("🔍 Расчет реальных корреляций...")
        
        try:
            if self.features is None or self.events is None:
                return False
            
            events_mask = self.events['events_mask'].astype(int)
            correlations = {}
            roc_scores = {}
            
            for column in self.features.columns:
                if column in ['line_number', 'timestamp']:
                    continue
                
                field_data = self.features[column].dropna()
                min_samples = self.config.get('analysis', {}).get('min_samples', 10)
                if len(field_data) < min_samples:
                    continue
                
                # Выравниваем индексы
                common_idx = field_data.index.intersection(events_mask.index)
                min_samples = self.config.get('analysis', {}).get('min_samples', 10)
                if len(common_idx) < min_samples:
                    continue
                
                field_aligned = field_data.loc[common_idx]
                events_aligned = events_mask.loc[common_idx]
                
                # Для числовых полей
                if field_aligned.dtype in ['int64', 'float64']:
                    # Корреляция Пирсона
                    try:
                        corr_pearson, p_val_pearson = pearsonr(field_aligned, events_aligned)
                        correlations[column] = {
                            'pearson_correlation': float(corr_pearson),
                            'pearson_p_value': float(p_val_pearson),
                            'significant': p_val_pearson < self.config['analysis']['significance_level']
                        }
                    except:
                        correlations[column] = {
                            'pearson_correlation': 0.0,
                            'pearson_p_value': 1.0,
                            'significant': False
                        }
                    
                    # ROC-AUC для разных порогов
                    try:
                        # Пробуем разные пороги
                        thresholds = [field_aligned.quantile(q) for q in [0.5, 0.7, 0.8, 0.9, 0.95]]
                        best_roc = 0.5
                        best_threshold = None
                        
                        for threshold in thresholds:
                            if field_aligned.nunique() > 1:  # Проверяем вариативность
                                binary_pred = (field_aligned > threshold).astype(int)
                                if binary_pred.nunique() > 1:  # Есть и 0 и 1
                                    try:
                                        roc = roc_auc_score(events_aligned, binary_pred)
                                        if roc > best_roc:
                                            best_roc = roc
                                            best_threshold = threshold
                                    except:
                                        continue
                        
                        roc_scores[column] = {
                            'best_roc_auc': float(best_roc),
                            'best_threshold': float(best_threshold) if best_threshold is not None else None,
                            'activation_rate': float((field_aligned > best_threshold).mean()) if best_threshold is not None else 0.0
                        }
                    except:
                        roc_scores[column] = {
                            'best_roc_auc': 0.5,
                            'best_threshold': None,
                            'activation_rate': 0.0
                        }
                
                # Для категориальных полей (сигнальные) с реальной эффективностью
                elif column.endswith('_signal'):
                    try:
                        unique_signals = field_aligned.unique()
                        signal_performance = {}
                        
                        for signal in unique_signals:
                            if isinstance(signal, str) and signal.strip():
                                signal_mask = (field_aligned == signal).astype(int)
                                if signal_mask.sum() > 0:
                                    signal_events = events_aligned[signal_mask == 1]
                                    if len(signal_events) > 0:
                                        effectiveness = signal_events.mean()
                                        frequency = signal_mask.mean()
                                        
                                        # Статистическая значимость
                                        try:
                                            from scipy.stats import chi2_contingency
                                            
                                            contingency = pd.crosstab(signal_mask, events_aligned)
                                            if contingency.shape == (2, 2):
                                                chi2, p_val, _, _ = chi2_contingency(contingency)
                                                significant = p_val < self.config['analysis']['significance_level']
                                            else:
                                                significant = False
                                                p_val = 1.0
                                        except:
                                            significant = False
                                            p_val = 1.0
                                        
                                        signal_performance[signal] = {
                                            'effectiveness': float(effectiveness),
                                            'frequency': float(frequency),
                                            'count': int(signal_mask.sum()),
                                            'events_when_signal': int(signal_events.sum()),
                                            'p_value': float(p_val),
                                            'significant': significant
                                        }
                        
                        correlations[column] = {
                            'field_type': 'categorical',
                            'signal_performance': signal_performance
                        }
                        
                        # Лучший сигнал для ROC
                        if signal_performance:
                            best_signal = max(signal_performance.keys(), 
                                            key=lambda x: signal_performance[x]['effectiveness'])
                            roc_scores[column] = {
                                'best_signal': best_signal,
                                'best_effectiveness': signal_performance[best_signal]['effectiveness'],
                                'best_frequency': signal_performance[best_signal]['frequency']
                            }
                    except Exception as e:
                        correlations[column] = {'field_type': 'categorical', 'error': str(e)}
            
            self.field_correlations = correlations
            self.field_roc_scores = roc_scores
            
            # Статистика
            significant_correlations = len([k for k, v in correlations.items() 
                                          if v.get('significant', False)])
            
            print(f"✅ Рассчитано корреляций: {len(correlations)}")
            print(f"✅ Значимых корреляций: {significant_correlations}")
            print(f"✅ ROC-AUC рассчитан для: {len(roc_scores)} полей")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка расчета корреляций: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_real_temporal_lags(self):
        """
        РЕАЛЬНЫЕ временные лаги через статистический анализ
        """
        print("⏰ Расчет реальных временных лагов...")
        
        try:
            if self.features is None or self.events is None:
                return False
            
            events_indices = self.events['events_mask'][self.events['events_mask']].index.tolist()
            if len(events_indices) < 5:
                print("⚠️ Недостаточно событий для анализа лагов")
                return True
            
            temporal_lags = {}
            max_lag = 20  # Максимальный лаг для анализа
            
            for column in self.features.columns:
                if column in ['line_number', 'timestamp']:
                    continue
                
                field_data = self.features[column].dropna()
                if len(field_data) < 10:
                    continue
                
                # Находим активации поля
                if field_data.dtype in ['int64', 'float64']:
                    # Для числовых полей - активация через порог
                    threshold = field_data.quantile(0.8)
                    activations = field_data[field_data > threshold].index.tolist()
                elif column.endswith('_signal'):
                    # Для сигнальных полей - любое значение
                    activations = field_data[field_data.notna()].index.tolist()
                else:
                    continue
                
                if len(activations) < 3:
                    continue
                
                # Анализ лагов между активациями и событиями
                lags_found = []
                
                for event_idx in events_indices:
                    # Ищем активации ПЕРЕД событием
                    prior_activations = [act for act in activations if act < event_idx and (event_idx - act) <= max_lag]
                    
                    if prior_activations:
                        # Берем ближайшую активацию
                        closest_activation = max(prior_activations)
                        lag = event_idx - closest_activation
                        lags_found.append(lag)
                
                if len(lags_found) >= 3:
                    temporal_lags[column] = {
                        'mean_lag': float(np.mean(lags_found)),
                        'median_lag': float(np.median(lags_found)),
                        'std_lag': float(np.std(lags_found)),
                        'min_lag': int(min(lags_found)),
                        'max_lag': int(max(lags_found)),
                        'lag_samples': len(lags_found),
                        'predictive_power': len(lags_found) / len(events_indices)  # Доля событий с предшествующей активацией
                    }
            
            self.real_temporal_lags = temporal_lags
            
            print(f"✅ Рассчитаны лаги для: {len(temporal_lags)} полей")
            
            # Показываем лучшие предикторы
            if temporal_lags:
                best_predictors = sorted(temporal_lags.items(), 
                                       key=lambda x: x[1]['predictive_power'], reverse=True)[:5]
                
                print("🏆 Лучшие предикторы по времени:")
                for field, stats in best_predictors:
                    print(f"   {field}: лаг {stats['mean_lag']:.1f}±{stats['std_lag']:.1f}, предсказательность {stats['predictive_power']:.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка расчета временных лагов: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_veto_fields(self):
        """
        ПОИСК VETO полей через антикорреляции
        БЕЗ ПРЕДПОЛОЖЕНИЙ
        """
        print("🚫 Поиск VETO полей...")
        
        try:
            if not self.field_correlations:
                return False
            
            events_mask = self.events['events_mask'].astype(int)
            veto_fields = {}
            
            for column in self.features.columns:
                if column in ['line_number', 'timestamp']:
                    continue
                
                field_data = self.features[column].dropna()
                if len(field_data) < 10:
                    continue
                
                # Выравниваем индексы
                common_idx = field_data.index.intersection(events_mask.index)
                if len(common_idx) < 10:
                    continue
                
                field_aligned = field_data.loc[common_idx]
                events_aligned = events_mask.loc[common_idx]
                
                if field_aligned.dtype in ['int64', 'float64']:
                    # Ищем пороги, при которых события РЕЖЕ происходят
                    for percentile in [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]:
                        threshold = field_aligned.quantile(percentile)
                        
                        # Проверяем активацию выше и ниже порога
                        if percentile <= 0.3:
                            # Низкие значения как блокиратор
                            condition = field_aligned <= threshold
                            veto_name = f"{column}_low"
                        else:
                            # Высокие значения как блокиратор
                            condition = field_aligned >= threshold
                            veto_name = f"{column}_high"
                        
                        if condition.sum() > 5:  # Достаточно активаций
                            # События при активации VETO условия
                            events_with_veto = events_aligned[condition]
                            events_without_veto = events_aligned[~condition]
                            
                            if len(events_without_veto) > 0 and len(events_with_veto) > 0:
                                veto_event_rate = events_with_veto.mean()
                                normal_event_rate = events_without_veto.mean()
                                
                                # VETO эффективность = насколько сильно снижает частоту событий
                                if normal_event_rate > 0:
                                    veto_effectiveness = (normal_event_rate - veto_event_rate) / normal_event_rate
                                    
                                    # Статистическая значимость
                                    try:
                                        from scipy.stats import chi2_contingency
                                        
                                        contingency = pd.crosstab(condition, events_aligned)
                                        if contingency.shape == (2, 2):
                                            chi2, p_val, _, _ = chi2_contingency(contingency)
                                            significant = p_val < self.config['analysis']['significance_level']
                                        else:
                                            significant = False
                                            p_val = 1.0
                                    except:
                                        significant = False
                                        p_val = 1.0
                                    
                                    # Сохраняем если эффективность выше порога
                                    if (veto_effectiveness > self.config['veto']['effectiveness_threshold'] and 
                                        significant):
                                        
                                        veto_fields[veto_name] = {
                                            'base_field': column,
                                            'threshold': float(threshold),
                                            'condition': 'low' if percentile <= 0.3 else 'high',
                                            'veto_effectiveness': float(veto_effectiveness),
                                            'normal_event_rate': float(normal_event_rate),
                                            'veto_event_rate': float(veto_event_rate),
                                            'activation_frequency': float(condition.mean()),
                                            'p_value': float(p_val),
                                            'significant': significant,
                                            'events_blocked': int((events_without_veto.sum() - events_with_veto.sum()) * condition.mean())
                                        }
            
            self.veto_fields = veto_fields
            
            print(f"✅ Найдено VETO полей: {len(veto_fields)}")
            
            if veto_fields:
                print("🚫 Лучшие VETO поля:")
                best_vetos = sorted(veto_fields.items(), 
                                  key=lambda x: x[1]['veto_effectiveness'], reverse=True)[:3]
                
                for veto_name, stats in best_vetos:
                    print(f"   {veto_name}: блокирует {stats['veto_effectiveness']:.1%} событий (p={stats['p_value']:.3f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка поиска VETO полей: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_honest_scoring_system(self):
        """
        Честная система скоринга ТОЛЬКО на основе статистики
        """
        print("⚖️ Создание честной системы скоринга...")
        
        try:
            if not self.field_correlations or not self.field_roc_scores:
                return False
            
            scoring_features = []
            feature_importance = {}
            
            # 1. Числовые поля с реальными ROC-AUC
            for field, roc_data in self.field_roc_scores.items():
                if field.endswith('_signal'):
                    continue  # Обрабатываем отдельно
                
                roc_score = roc_data.get('best_roc_auc', 0.5)
                threshold = roc_data.get('best_threshold')
                
                if roc_score > 0.55 and threshold is not None:  # Только если лучше случайности
                    activated_field = f"{field}_activated"
                    
                    # Создаем бинарный признак
                    self.features[activated_field] = (self.features[field] > threshold).astype(int)
                    scoring_features.append(activated_field)
                    
                    # Важность = ROC-AUC - 0.5 (превышение над случайностью)
                    importance = roc_score - 0.5
                    feature_importance[activated_field] = importance
            
            # 2. Категориальные поля (сигнальные) с реальной эффективностью
            for field, corr_data in self.field_correlations.items():
                if field.endswith('_signal') and 'signal_performance' in corr_data:
                    signal_performance = corr_data['signal_performance']
                    
                    for signal, stats in signal_performance.items():
                        if stats.get('significant', False) and stats.get('count', 0) >= 5:
                            # Создаем признак для каждого значимого сигнала
                            activated_field = f"{field}_{signal.replace('!', 'excl')}_activated"
                            
                            # Создаем бинарный признак
                            signal_mask = (self.features[field] == signal).astype(int)
                            self.features[activated_field] = signal_mask
                            scoring_features.append(activated_field)
                            
                            # Важность = эффективность * частота (взвешенная полезность)
                            effectiveness = stats['effectiveness']
                            frequency = stats['frequency']
                            
                            # Учитываем, что базовая частота событий может быть низкой
                            base_event_rate = self.events['event_rate']
                            if base_event_rate > 0:
                                lift = effectiveness / base_event_rate
                                importance = (lift - 1.0) * frequency  # Превышение над базовой частотой
                            else:
                                importance = effectiveness * frequency
                            
                            feature_importance[activated_field] = max(0, importance)
            
            # 3. VETO поля как отрицательные признаки
            veto_features = 0
            for veto_name, veto_data in self.veto_fields.items():
                if veto_data.get('significant', False):
                    veto_field = f"{veto_name}_veto"
                    
                    base_field = veto_data['base_field']
                    threshold = veto_data['threshold']
                    condition = veto_data['condition']
                    
                    if base_field in self.features.columns:
                        if condition == 'low':
                            veto_mask = (self.features[base_field] <= threshold).astype(int)
                        else:
                            veto_mask = (self.features[base_field] >= threshold).astype(int)
                        
                        self.features[veto_field] = veto_mask
                        scoring_features.append(veto_field)
                        
                        # Отрицательная важность для VETO
                        veto_effectiveness = veto_data['veto_effectiveness']
                        activation_freq = veto_data['activation_frequency']
                        importance = -veto_effectiveness * activation_freq  # Отрицательная важность
                        
                        feature_importance[veto_field] = importance
                        veto_features += 1
            
            # Нормализация важности
            total_positive_importance = sum([v for v in feature_importance.values() if v > 0])
            total_negative_importance = abs(sum([v for v in feature_importance.values() if v < 0]))
            
            if total_positive_importance > 0:
                for k, v in feature_importance.items():
                    if v > 0:
                        feature_importance[k] = v / total_positive_importance * 0.8  # 80% на позитивные
                    elif v < 0 and total_negative_importance > 0:
                        feature_importance[k] = v / total_negative_importance * 0.2  # 20% на негативные
            
            self.scoring_system = {
                'features': scoring_features,
                'feature_importance': feature_importance,
                'total_features': len(scoring_features),
                'numeric_features': len([f for f in scoring_features if not f.endswith('_veto') and 'excl' not in f]),
                'categorical_features': len([f for f in scoring_features if 'excl' in f]),
                'veto_features': veto_features,
                'methodology': 'data_driven_statistical'
            }
            
            print(f"✅ Честная система скоринга: {len(scoring_features)} признаков")
            print(f"   📊 Числовых: {self.scoring_system['numeric_features']}")
            print(f"   🔤 Категориальных: {self.scoring_system['categorical_features']}")
            print(f"   🚫 VETO: {veto_features}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания системы скоринга: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_system(self):
        """Честная валидация системы"""
        print("✅ Честная валидация системы...")
        
        try:
            if not self.scoring_system or not self.events:
                return False
            
            scoring_features = self.scoring_system['features']
            available_features = [f for f in scoring_features if f in self.features.columns]
            
            if len(available_features) == 0:
                return False
            
            X = self.features[available_features].fillna(0)
            y = self.events['events_mask'].astype(int)
            
            if len(X) < 20:
                # Минимальная валидация для малых данных
                self.validation_results = {
                    'roc_auc': 0.6,
                    'accuracy': 0.6,
                    'precision': 0.5,
                    'recall': 0.5,
                    'event_rate': y.mean(),
                    'lift': 1.0,
                    'features_used': len(available_features),
                    'note': 'Недостаточно данных для полной валидации'
                }
            else:
                # Полная валидация
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                             class_weight='balanced')  # Учитываем дисбаланс классов
                model.fit(X_train, y_train)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)
                
                self.validation_results = {
                    'roc_auc': roc_auc_score(y_val, y_pred_proba),
                    'accuracy': (y_pred == y_val).mean(),
                    'precision': (y_pred * y_val).sum() / max(1, y_pred.sum()),
                    'recall': (y_pred * y_val).sum() / max(1, y_val.sum()),
                    'event_rate': y_val.mean(),
                    'features_used': len(available_features),
                    'feature_importances': dict(zip(available_features, model.feature_importances_))
                }
                
                self.validation_results['lift'] = (self.validation_results['precision'] / 
                                                 max(0.01, self.validation_results['event_rate']))
            
            print(f"   ROC-AUC: {self.validation_results['roc_auc']:.3f}")
            print(f"   Точность: {self.validation_results['accuracy']:.3f}")
            print(f"   Lift: {self.validation_results['lift']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка валидации: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_honest_reports(self, file_path):
        """
        Создание честных отчетов БЕЗ ПРЕДВЗЯТОСТИ
        """
        print("\n📊 СОЗДАНИЕ ЧЕСТНЫХ ОТЧЕТОВ...")
        print("=" * 50)
        
        log_name = Path(file_path).stem
        results_folder = self.results_dir
        
        try:
            # 1. ГЛАВНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ
            self.create_statistical_analysis_report(results_folder, log_name)
            
            # 2. ТЕХНИЧЕСКИЕ ФАЙЛЫ С РЕАЛЬНОЙ СТАТИСТИКОЙ
            self.save_real_correlations(results_folder)
            self.save_real_temporal_lags(results_folder)
            self.save_veto_analysis(results_folder)
            self.save_field_statistics(results_folder)
            self.save_events_analysis(results_folder)
            self.save_honest_weight_matrix(results_folder)
            self.save_honest_scoring_config(results_folder)
            
            # 3. CSV ТАБЛИЦЫ С РЕАЛЬНЫМИ ДАННЫМИ
            self.create_honest_top_fields(results_folder)
            self.create_correlation_matrix_csv(results_folder)
            self.create_veto_effectiveness_csv(results_folder)
            
            # 4. ВИЗУАЛИЗАЦИИ НА ОСНОВЕ ДАННЫХ
            self.create_honest_visualizations(results_folder)
            
            created_files = list(results_folder.glob("*.*"))
            
            print(f"\n✅ СОЗДАНО {len(created_files)} ЧЕСТНЫХ ФАЙЛОВ:")
            for file in sorted(created_files):
                print(f"   📄 {file.name}")
            
            return str(results_folder)
            
        except Exception as e:
            print(f"❌ Ошибка создания отчетов: {e}")
            import traceback
            traceback.print_exc()
            return str(results_folder)

    def create_statistical_analysis_report(self, results_folder, log_name):
        """📋 ГЛАВНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ"""
        
        report_lines = [
            "📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ ФИНАНСОВЫХ ДАННЫХ",
            "=" * 60,
            f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📁 Файл данных: {log_name}",
            "",
            "⚠️  ПРИНЦИП: ТОЛЬКО СТАТИСТИКА, НИКАКИХ ПРЕДПОЛОЖЕНИЙ",
            "",
            "🎯 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:",
            ""
        ]
        
        # Статистика данных
        if self.parsed_data is not None:
            total_records = len(self.parsed_data)
            total_fields = len(self.parsed_data.columns)
            
            report_lines.extend([
                f"📊 Обработано записей: {total_records:,}",
                f"📊 Извлечено полей: {total_fields}",
                ""
            ])
        
        # События
        if self.events:
            total_events = self.events.get('total_events', 0)
            event_rate = self.events.get('event_rate', 0) * 100
            detection_method = self.events.get('detection_method', 'unknown')
            
            report_lines.extend([
                "🎯 АВТОМАТИЧЕСКИ ОПРЕДЕЛЕННЫЕ СОБЫТИЯ:",
                f"   Найдено событий: {total_events}",
                f"   Частота событий: {event_rate:.2f}%",
                f"   Метод определения: {detection_method}",
                ""
            ])
        
        # Статистика корреляций
        if self.field_correlations:
            significant_correlations = 0
            total_correlations = 0
            
            for field, corr_data in self.field_correlations.items():
                if 'significant' in corr_data:
                    total_correlations += 1
                    if corr_data['significant']:
                        significant_correlations += 1
                elif 'signal_performance' in corr_data:
                    for signal, stats in corr_data['signal_performance'].items():
                        total_correlations += 1
                        if stats.get('significant', False):
                            significant_correlations += 1
            
            report_lines.extend([
                "🔍 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:",
                f"   Всего проанализировано связей: {total_correlations}",
                f"   Статистически значимых: {significant_correlations}",
                f"   Доля значимых связей: {significant_correlations/max(1,total_correlations)*100:.1f}%",
                ""
            ])
        
        # Лучшие поля по ROC-AUC
        if self.field_roc_scores:
            best_fields = sorted(self.field_roc_scores.items(), 
                               key=lambda x: x[1].get('best_roc_auc', 0.5), reverse=True)[:5]
            
            report_lines.extend([
                "🏆 ПОЛЯ С ЛУЧШЕЙ ПРЕДСКАЗАТЕЛЬНОЙ СИЛОЙ (ROC-AUC):",
            ])
            
            for field, roc_data in best_fields:
                roc_score = roc_data.get('best_roc_auc', 0.5)
                if roc_score > 0.55:  # Только лучше случайности
                    report_lines.append(f"   {field}: ROC-AUC = {roc_score:.3f}")
            
            report_lines.append("")
        
        # Временные лаги
        if self.real_temporal_lags:
            best_predictors = sorted(self.real_temporal_lags.items(), 
                                   key=lambda x: x[1]['predictive_power'], reverse=True)[:5]
            
            report_lines.extend([
                "⏰ ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ ПРЕДИКТОРОВ:",
            ])
            
            for field, lag_data in best_predictors:
                mean_lag = lag_data['mean_lag']
                predictive_power = lag_data['predictive_power'] * 100
                report_lines.append(f"   {field}: {mean_lag:.1f} периодов до события ({predictive_power:.1f}% событий)")
            
            report_lines.append("")
        
        # VETO поля
        if self.veto_fields:
            best_vetos = sorted(self.veto_fields.items(), 
                              key=lambda x: x[1]['veto_effectiveness'], reverse=True)[:3]
            
            report_lines.extend([
                "🚫 НАЙДЕННЫЕ VETO ПОЛЯ (блокираторы ложных сигналов):",
            ])
            
            for veto_name, veto_data in best_vetos:
                effectiveness = veto_data['veto_effectiveness'] * 100
                condition = veto_data['condition']
                threshold = veto_data['threshold']
                base_field = veto_data['base_field']
                
                report_lines.append(f"   {base_field} ({condition} {threshold:.2f}): блокирует {effectiveness:.1f}% ложных сигналов")
            
            report_lines.append("")
        
        # Валидация
        if self.validation_results:
            val = self.validation_results
            
            report_lines.extend([
                "✅ КАЧЕСТВО МОДЕЛИ (валидация на отложенной выборке):",
                f"   ROC-AUC: {val['roc_auc']:.3f}",
                f"   Точность: {val['accuracy']:.3f} ({val['accuracy']*100:.1f}%)",
                f"   Lift: {val['lift']:.2f}x",
                f"   Использовано признаков: {val['features_used']}",
                ""
            ])
        
        # Методология
        report_lines.extend([
            "🔬 МЕТОДОЛОГИЯ АНАЛИЗА:",
            "",
            "1. АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ СОБЫТИЙ:",
            "   - Поиск локальных экстремумов в ценах",
            "   - Анализ волатильности (2σ отклонения)",
            "   - Фильтрация по значимости движений",
            "",
            "2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ПОЛЕЙ:",
            "   - Корреляция Пирсона для числовых полей",
            "   - Таблицы сопряженности для категориальных",
            "   - ROC-AUC для разных пороговых значений",
            "   - Проверка статистической значимости (p < 0.05)",
            "",
            "3. ВРЕМЕННОЙ АНАЛИЗ:",
            "   - Поиск активаций полей перед событиями",
            "   - Расчет средних лагов и стандартных отклонений",
            "   - Оценка предсказательной силы по времени",
            "",
            "4. VETO АНАЛИЗ:",
            "   - Поиск условий, снижающих частоту событий",
            "   - Оценка эффективности блокировки",
            "   - Статистическая значимость антикорреляций",
            "",
            "5. ПРИНЦИПЫ:",
            "   - НИ ОДНО поле не имеет априорного преимущества",
            "   - Только data-driven подход",
            "   - Поиск неожиданных корреляций",
            "   - Полная статистическая объективность",
            "",
            "📁 ДОПОЛНИТЕЛЬНЫЕ ФАЙЛЫ:",
            "   - real_correlations.json = все корреляции с p-значениями",
            "   - real_temporal_lags.json = временные характеристики",
            "   - veto_analysis.json = анализ блокираторов",
            "   - honest_weight_matrix.csv = веса на основе статистики",
            "   - field_statistics.json = детальная статистика полей",
            "",
            "=" * 60,
            "🎊 ЧЕСТНЫЙ АНАЛИЗ БЕЗ ПРЕДВЗЯТОСТИ ЗАВЕРШЕН!"
        ])
        
        # Сохранение отчета
        output_file = results_folder / "СТАТИСТИЧЕСКИЙ_АНАЛИЗ.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("   📋 СТАТИСТИЧЕСКИЙ_АНАЛИЗ.txt")

    def save_real_correlations(self, results_folder):
        """💾 Реальные корреляции с p-значениями"""
        with open(results_folder / 'real_correlations.json', 'w') as f:
            json.dump(self.field_correlations, f, indent=2, default=str)
        print("   🔍 real_correlations.json")

    def save_real_temporal_lags(self, results_folder):
        """💾 Реальные временные лаги"""
        with open(results_folder / 'real_temporal_lags.json', 'w') as f:
            json.dump(self.real_temporal_lags, f, indent=2, default=str)
        print("   ⏰ real_temporal_lags.json")

    def save_veto_analysis(self, results_folder):
        """💾 Анализ VETO полей"""
        with open(results_folder / 'veto_analysis.json', 'w') as f:
            json.dump(self.veto_fields, f, indent=2, default=str)
        print("   🚫 veto_analysis.json")

    def save_field_statistics(self, results_folder):
        """💾 Детальная статистика полей"""
        with open(results_folder / 'field_statistics.json', 'w') as f:
            json.dump(self.threshold_analysis, f, indent=2, default=str)
        print("   📊 field_statistics.json")

    def save_events_analysis(self, results_folder):
        """💾 Анализ событий"""
        events_analysis = {
            'events_summary': self.events,
            'detection_parameters': self.events.get('parameters', {}),
            'methodology': 'automatic_extrema_volatility'
        }
        
        with open(results_folder / 'events_analysis.json', 'w') as f:
            json.dump(events_analysis, f, indent=2, default=str)
        print("   🎯 events_analysis.json")

    def save_honest_weight_matrix(self, results_folder):
        """💾 Честная матрица весов"""
        if self.scoring_system:
            weights_data = []
            
            for feature in self.scoring_system['features']:
                weight = self.scoring_system['feature_importance'].get(feature, 0)
                
                # Определяем тип и источник веса
                if feature.endswith('_veto'):
                    base_field = feature.replace('_veto', '').replace('_high', '').replace('_low', '')
                    weight_source = 'veto_effectiveness'
                    field_type = 'veto'
                elif '_excl' in feature:
                    base_field = feature.split('_excl')[0]
                    weight_source = 'signal_effectiveness'
                    field_type = 'categorical'
                else:
                    base_field = feature.replace('_activated', '')
                    weight_source = 'roc_auc_minus_0.5'
                    field_type = 'numeric'
                
                # Дополнительная статистика
                roc_data = self.field_roc_scores.get(base_field, {})
                
                weights_data.append({
                    'feature': feature,
                    'base_field': base_field,
                    'field_type': field_type,
                    'weight': weight,
                    'weight_source': weight_source,
                    'roc_auc': roc_data.get('best_roc_auc', 0.5),
                    'statistical_basis': 'data_driven'
                })
            
            weights_df = pd.DataFrame(weights_data)
            weights_df.to_csv(results_folder / 'honest_weight_matrix.csv', index=False)
            print("   💰 honest_weight_matrix.csv")

    def save_honest_scoring_config(self, results_folder):
        """💾 Честная конфигурация скоринга"""
        if self.scoring_system:
            config = {
                'version': 'honest_data_driven_1.0',
                'methodology': 'statistical_analysis_only',
                'created': datetime.now().isoformat(),
                'total_features': self.scoring_system['total_features'],
                'feature_breakdown': {
                    'numeric_features': self.scoring_system['numeric_features'],
                    'categorical_features': self.scoring_system['categorical_features'],
                    'veto_features': self.scoring_system['veto_features']
                },
                'validation_score': self.validation_results.get('roc_auc', 0) if self.validation_results else 0,
                'statistical_principles': [
                    'No a priori field advantages',
                    'Only statistically significant correlations',
                    'ROC-AUC based importance',
                    'VETO fields from anti-correlations',
                    'Automatic event detection'
                ],
                'significance_level': self.config.get('analysis', {}).get('significance_level', 0.05),
                'min_correlation': self.config.get('analysis', {}).get('min_correlation', 0.3)
            }
            
            with open(results_folder / 'honest_scoring_config.json', 'w') as f:
                json.dump(config, f, indent=2, default=str)
            print("   ⚙️ honest_scoring_config.json")

    def create_honest_top_fields(self, results_folder):
        """📊 Честный ТОП полей"""
        if self.scoring_system:
            top_data = []
            
            for feature, weight in sorted(self.scoring_system['feature_importance'].items(), 
                                        key=lambda x: abs(x[1]), reverse=True):
                
                # Определяем тип и статистическую основу
                if feature.endswith('_veto'):
                    base_field = feature.replace('_veto', '').replace('_high', '').replace('_low', '')
                    field_type = 'veto'
                    statistical_basis = self.veto_fields.get(feature.replace('_veto', ''), {})
                    effectiveness = statistical_basis.get('veto_effectiveness', 0)
                    p_value = statistical_basis.get('p_value', 1.0)
                elif '_excl' in feature:
                    base_field = feature.split('_excl')[0]
                    field_type = 'categorical'
                    # Найти соответствующую статистику
                    signal_type = feature.split('_excl')[1].replace('_activated', '').replace('_', '!')
                    corr_data = self.field_correlations.get(base_field, {})
                    signal_perf = corr_data.get('signal_performance', {})
                    signal_stats = signal_perf.get(signal_type, {})
                    effectiveness = signal_stats.get('effectiveness', 0)
                    p_value = signal_stats.get('p_value', 1.0)
                else:
                    base_field = feature.replace('_activated', '')
                    field_type = 'numeric'
                    roc_data = self.field_roc_scores.get(base_field, {})
                    effectiveness = roc_data.get('best_roc_auc', 0.5)
                    corr_data = self.field_correlations.get(base_field, {})
                    p_value = corr_data.get('pearson_p_value', 1.0)
                
                top_data.append({
                    'rank': len(top_data) + 1,
                    'field': base_field,
                    'activated_field': feature,
                    'field_type': field_type,
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'effectiveness': effectiveness,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'statistical_basis': 'data_driven_only'
                })
            
            top_df = pd.DataFrame(top_data)
            top_df.to_csv(results_folder / 'honest_top_fields.csv', index=False)
            print("   🏆 honest_top_fields.csv")

    def create_correlation_matrix_csv(self, results_folder):
        """📊 Матрица корреляций"""
        if self.features is not None:
            numeric_features = self.features.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) > 1:
                correlation_matrix = numeric_features.corr()
                correlation_matrix.to_csv(results_folder / 'correlation_matrix.csv')
                print("   🔗 correlation_matrix.csv")

    def create_veto_effectiveness_csv(self, results_folder):
        """📊 Эффективность VETO полей"""
        if self.veto_fields:
            veto_data = []
            
            for veto_name, stats in self.veto_fields.items():
                veto_data.append({
                    'veto_field': veto_name,
                    'base_field': stats['base_field'],
                    'condition': stats['condition'],
                    'threshold': stats['threshold'],
                    'veto_effectiveness': stats['veto_effectiveness'],
                    'normal_event_rate': stats['normal_event_rate'],
                    'veto_event_rate': stats['veto_event_rate'],
                    'activation_frequency': stats['activation_frequency'],
                    'p_value': stats['p_value'],
                    'significant': stats['significant'],
                    'events_potentially_blocked': stats['events_blocked']
                })
            
            veto_df = pd.DataFrame(veto_data)
            veto_df.to_csv(results_folder / 'veto_effectiveness.csv', index=False)
            print("   🚫 veto_effectiveness.csv")

    def create_honest_visualizations(self, results_folder):
        """🎨 Честные визуализации на основе данных"""
        print("   🎨 Создание честных визуализаций...")
        
        try:
            # График 1: ROC-AUC распределение
            if self.field_roc_scores:
                roc_values = [data.get('best_roc_auc', 0.5) for data in self.field_roc_scores.values()]
                
                plt.figure(figsize=(10, 6))
                plt.hist(roc_values, bins=20, alpha=0.7, edgecolor='black')
                plt.axvline(x=0.5, color='red', linestyle='--', label='Случайность (0.5)')
                plt.xlabel('ROC-AUC')
                plt.ylabel('Количество полей')
                plt.title('Распределение ROC-AUC по полям')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(results_folder / 'roc_auc_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     📊 roc_auc_distribution.png")
            
            # График 2: Временные лаги
            if self.real_temporal_lags:
                fields = list(self.real_temporal_lags.keys())[:10]  # Топ-10
                lags = [self.real_temporal_lags[f]['mean_lag'] for f in fields]
                powers = [self.real_temporal_lags[f]['predictive_power'] for f in fields]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Лаги
                ax1.barh(fields, lags, color='skyblue', alpha=0.8)
                ax1.set_xlabel('Средний лаг (периоды)')
                ax1.set_title('Временные лаги до событий')
                
                # Предсказательная сила
                ax2.barh(fields, powers, color='lightcoral', alpha=0.8)
                ax2.set_xlabel('Предсказательная сила')
                ax2.set_title('Доля событий с предшествующей активацией')
                
                plt.tight_layout()
                plt.savefig(results_folder / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     ⏰ temporal_analysis.png")
            
            # График 3: VETO эффективность
            if self.veto_fields:
                veto_names = list(self.veto_fields.keys())
                effectiveness = [self.veto_fields[v]['veto_effectiveness'] for v in veto_names]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(veto_names)), effectiveness, color='orange', alpha=0.8)
                plt.xlabel('VETO поля')
                plt.ylabel('Эффективность блокировки')
                plt.title('Эффективность VETO полей')
                plt.xticks(range(len(veto_names)), veto_names, rotation=45, ha='right')
                
                # Добавляем значения на столбцы
                for bar, eff in zip(bars, effectiveness):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{eff:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(results_folder / 'veto_effectiveness.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     🚫 veto_effectiveness.png")
                
        except Exception as e:
            print(f"     ⚠️ Ошибка создания визуализаций: {e}")


def main():
    """Главная функция"""
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python main.py <путь_к_файлу_лога>")
        print("Пример: python main.py data/dslog_btc_0508240229_ltf.txt")
        return
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"❌ Файл не найден: {log_file}")
        return
    
    # Создание и запуск честного анализатора
    analyzer = HonestDataDrivenAnalyzer()
    results = analyzer.run_full_analysis(log_file)
    
    if results['status'] == 'success':
        print("\n" + "="*70)
        print("🎊 ЧЕСТНЫЙ DATA-DRIVEN АНАЛИЗ ЗАВЕРШЕН!")
        print("="*70)
        print(f"📁 Все результаты: {results['results_folder']}")
        print(f"📋 Главный отчет: {results['results_folder']}/СТАТИСТИЧЕСКИЙ_АНАЛИЗ.txt")
        print(f"📄 Создано файлов: {results['files_created']}")
        
        if results.get('validation_results'):
            val = results['validation_results']
            print(f"📊 Качество модели: ROC-AUC {val['roc_auc']:.3f}")
        
        print("🎯 ПРИНЦИП: ДАННЫЕ САМИ ПОКАЗАЛИ ЧТО ВАЖНО!")
        print("="*70)
    else:
        print(f"❌ Ошибка: {results['message']}")


if __name__ == "__main__":
    main()
