#!/usr/bin/env python3
"""
ИСПРАВЛЕННЫЙ ПАРСЕР ЛОГОВ - ИЗВЛЕЧЕНИЕ ВСЕХ ИНДИКАТОРНЫХ ПОЛЕЙ
Исправляет критическую ошибку с отрицательными значениями (ef2--7.19, ze2--4.12)

Извлекает ВСЕ поля из сырого лога:
- nw (сигналы !!, !!!) 
- ef (energy factor) с отрицательными значениями
- as (accumulated signal)
- vc (volatility composite)
- ze (zero crossing) с отрицательными значениями
- maz, cvz, dz, rz, mz (sigma поля)
- co, ro, mo, do, so (momentum поля)
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class AdvancedLogParser:
    """
    ИСПРАВЛЕННЫЙ парсер для извлечения ВСЕХ полей из финансовых логов
    """
    
    def __init__(self):
        """Инициализация парсера"""
        self.field_patterns = self._create_field_patterns()
        self.metadata_patterns = self._create_metadata_patterns()
        self.parsed_data = []
        self.field_statistics = {}
        
    def _create_field_patterns(self) -> Dict[str, str]:
        """Создание ИСПРАВЛЕННЫХ паттернов для извлечения полей"""
        
        # Определяем точные списки полей из ТЗ
        self.ltf_field_prefixes = {
            'rd', 'md', 'cd', 'cmd', 'macd', 'cvd', 'dd', 'ed', 'sd', 
            'ro', 'mo', 'co', 'cz', 'do', 'so', 'rz', 'mz', 'ciz', 'sz', 
            'dz', 'cvz', 'maz', 'ef', 'vc', 'ze', 'nw', 'as', 'vw'
        }
        
        self.htf_field_prefixes = {
            'rd', 'md', 'cd', 'cmd', 'macd', 'od', 'dd', 'cvd', 'drd', 'ad',
            'ed', 'hd', 'sd', 'ro', 'mo', 'co', 'cz', 'do', 'ae', 'so',
            'rz', 'mz', 'ciz', 'sz', 'dz', 'ef', 'wv', 'vc', 'ze', 'nw',
            'dz', 'cvz', 'maz', 'oz'
        }
        
        self.special_htf_fields = {'bs', 'wa', 'pd'}
        
        # Суффиксы для определения LTF/HTF
        self.ltf_suffixes = {'2', '5', '15', '30'}
        self.htf_suffixes = {'1h', '4h', '1d', '1w'}
        
        patterns = {
            # ИСПРАВЛЕННЫЙ универсальный паттерн для всех полей данных
            # Обрабатывает: ef2--7.19, nw2-!!, as2-3.33, ze2--4.12
            'universal_field': r'([a-zA-Z]+)(\d+|1h|4h|1d|1w)-((?:!+)|(?:--?\d+(?:\.\d+)?(?:%)?)|(?:-?\d+(?:\.\d+)?(?:%)?))',
            
            # Специальные HTF поля без суффиксов
            'special_htf': r'\b(bs|wa|pd)\b(?:\s+([^\s,|]+))?',
            
            # Контекст зрелости (progress поля)
            'progress_ltf': r'p(\d+)-(-?\d+(?:\.\d+)?)',
            'progress_htf': r'p(1h|4h|1d|1w)-(-?\d+(?:\.\d+)?)',
        }
        return patterns
    
    def _create_metadata_patterns(self) -> Dict[str, str]:
        """Паттерны для метаданных свечей (вторичные данные)"""
        return {
            'timestamp': r'\[([^\]]+)\]',
            'ohlc': r'o:([0-9.]+)\|h:([0-9.]+)\|l:([0-9.]+)\|c:([0-9.]+)',
            'volume': r'\|([0-9.]+K)\|',
            'range': r'rng:([0-9.]+)',
            'candle_type': r'\|(NORMAL|BIG_BODY|DOJI|HAMMER|PIN_TOP|FLAT)\|',
            'color': r'\|(RED|GREEN)\|',
            'change_24h': r'(-?\d+(?:\.\d+)?)%_24h'
        }
    
    def parse_log_file(self, file_path: str) -> pd.DataFrame:
        """
        Парсинг файла лога с извлечением ВСЕХ полей
        
        Args:
            file_path: путь к файлу лога
            
        Returns:
            DataFrame с извлеченными полями
        """
        print(f"🔍 Анализ лога: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📋 Найдено {len(lines)} строк")
        
        parsed_records = []
        
        for i, line in enumerate(lines):
            try:
                record = self._parse_single_line(line.strip(), i)
                if record:
                    parsed_records.append(record)
                    
                # Показываем прогресс
                if (i + 1) % 1000 == 0:
                    print(f"   Обработано: {i + 1}/{len(lines)} строк")
                    
            except Exception as e:
                print(f"⚠️ Ошибка в строке {i}: {str(e)[:100]}")
                continue
        
        if not parsed_records:
            print("❌ Не удалось извлечь данные из лога")
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_records)
        
        # Статистика извлеченных полей
        self._generate_parsing_statistics(df)
        
        print(f"✅ Извлечено {len(df)} записей с {len(df.columns)} полями")
        return df
    
    def _parse_single_line(self, line: str, line_num: int) -> Optional[Dict]:
        """Парсинг одной строки лога"""
        if not line or line.startswith('#'):
            return None
        
        record = {'line_number': line_num, 'raw_line': line}
        
        # Извлечение метаданных
        metadata = self._extract_metadata(line)
        record.update(metadata)
        
        # Извлечение ВСЕХ индикаторных полей
        indicator_fields = self._extract_all_indicator_fields(line)
        record.update(indicator_fields)
        
        return record
    
    def _extract_metadata(self, line: str) -> Dict:
        """Извлечение метаданных свечи"""
        metadata = {}
        
        # Timestamp
        ts_match = re.search(self.metadata_patterns['timestamp'], line)
        if ts_match:
            metadata['timestamp'] = ts_match.group(1)
        
        # OHLC
        ohlc_match = re.search(self.metadata_patterns['ohlc'], line)
        if ohlc_match:
            metadata['open'] = float(ohlc_match.group(1))
            metadata['high'] = float(ohlc_match.group(2))
            metadata['low'] = float(ohlc_match.group(3))
            metadata['close'] = float(ohlc_match.group(4))
        
        # Volume
        vol_match = re.search(r'\|([0-9.]+)K\|', line)
        if vol_match:
            metadata['volume'] = float(vol_match.group(1))
        
        # Range
        rng_match = re.search(r'rng:([0-9.]+)', line)
        if rng_match:
            metadata['range'] = float(rng_match.group(1))
        
        # Candle type and color
        if 'RED' in line:
            metadata['candle_color'] = 'RED'
        elif 'GREEN' in line:
            metadata['candle_color'] = 'GREEN'
            
        if 'BIG_BODY' in line:
            metadata['candle_type'] = 'BIG_BODY'
        elif 'DOJI' in line:
            metadata['candle_type'] = 'DOJI'
        elif 'HAMMER' in line:
            metadata['candle_type'] = 'HAMMER'
        elif 'PIN_TOP' in line:
            metadata['candle_type'] = 'PIN_TOP'
        elif 'FLAT' in line:
            metadata['candle_type'] = 'FLAT'
        else:
            metadata['candle_type'] = 'NORMAL'
        
        return metadata
    
    def _extract_all_indicator_fields(self, line: str) -> Dict:
        """ИСПРАВЛЕННОЕ универсальное извлечение всех полей данных"""
        fields = {}
        
        # Универсальное извлечение всех полей формата prefix+suffix-value
        universal_matches = re.findall(self.field_patterns['universal_field'], line)
        
        for prefix, suffix, value in universal_matches:
            field_name = f"{prefix}{suffix}"
            
            # Определяем тип поля (LTF/HTF) по суффиксу
            is_ltf = suffix in self.ltf_suffixes
            is_htf = suffix in self.htf_suffixes
            
            # Проверяем что префикс существует в соответствующих списках
            valid_ltf = is_ltf and prefix in self.ltf_field_prefixes
            valid_htf = is_htf and prefix in self.htf_field_prefixes
            
            if valid_ltf or valid_htf:
                # Обработка специальных сигналов nw (!!!, !!)
                if prefix == 'nw' and '!' in value:
                    fields[field_name] = len(value)  # Количество !
                    fields[f"{field_name}_signal"] = value  # Сам сигнал
                
                # Обработка числовых значений (включая отрицательные с двойными дефисами)
                elif self._is_numeric_value(value):
                    try:
                        # Конвертируем значение в число
                        numeric_value = self._parse_numeric_value(value)
                        fields[field_name] = numeric_value
                        
                        # Маркировка типа поля
                        if is_ltf:
                            fields[f"{field_name}_type"] = 'LTF'
                        elif is_htf:
                            fields[f"{field_name}_type"] = 'HTF'
                            
                    except ValueError:
                        # Если не число, сохраняем как строку
                        fields[field_name] = value
                
                # Обработка текстовых значений
                else:
                    fields[field_name] = value
        
        # Специальные HTF поля (bs, wa, pd)
        special_matches = re.findall(self.field_patterns['special_htf'], line)
        for field, value in special_matches:
            if field in self.special_htf_fields:
                fields[field] = value if value else 1  # 1 если просто присутствует
                fields[f"{field}_type"] = 'HTF_SPECIAL'
        
        # Контекст зрелости (progress поля)
        # LTF progress
        p_ltf_matches = re.findall(self.field_patterns['progress_ltf'], line)
        for suffix, value in p_ltf_matches:
            field_name = f"p{suffix}"
            fields[field_name] = float(value)
            fields[f"{field_name}_type"] = 'LTF_PROGRESS'
        
        # HTF progress 
        p_htf_matches = re.findall(self.field_patterns['progress_htf'], line)
        for suffix, value in p_htf_matches:
            field_name = f"p{suffix}"
            fields[field_name] = float(value)
            fields[f"{field_name}_type"] = 'HTF_PROGRESS'
        
        return fields
    
    def _is_numeric_value(self, value: str) -> bool:
        """Проверяет является ли значение числовым (включая отрицательные с --)"""
        # Убираем % если есть
        clean_value = value.replace('%', '')
        
        # Проверяем паттерны числовых значений
        numeric_patterns = [
            r'^-?\d+(?:\.\d+)?$',      # 3.33, -4.12
            r'^--\d+(?:\.\d+)?$',      # --7.19, --4.12
        ]
        
        for pattern in numeric_patterns:
            if re.match(pattern, clean_value):
                return True
        
        return False
    
    def _parse_numeric_value(self, value: str) -> float:
        """Парсит числовое значение из строки"""
        # Убираем % если есть
        clean_value = value.replace('%', '')
        
        # Обрабатываем отрицательные значения с двойными дефисами
        if clean_value.startswith('--'):
            # --7.19 → -7.19
            return -float(clean_value[2:])
        else:
            # 3.33, -4.12
            return float(clean_value)
    
    def _generate_parsing_statistics(self, df: pd.DataFrame):
        """Генерация статистики извлеченных полей"""
        print("\n📊 СТАТИСТИКА ИЗВЛЕЧЕННЫХ ПОЛЕЙ:")
        
        # Подсчет полей по группам
        field_groups = {
            'nw_fields': [col for col in df.columns if col.startswith('nw') and not col.endswith('_signal') and not col.endswith('_type')],
            'ef_fields': [col for col in df.columns if col.startswith('ef') and not col.endswith('_type')],
            'as_fields': [col for col in df.columns if col.startswith('as') and not col.endswith('_type')],
            'vc_fields': [col for col in df.columns if col.startswith('vc') and not col.endswith('_type')],
            'ze_fields': [col for col in df.columns if col.startswith('ze') and not col.endswith('_type')],
            'sigma_fields': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['rz', 'mz', 'cz', 'dz', 'cvz', 'maz', 'ciz', 'sz']) and not col.endswith('_type')],
            'momentum_fields': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['co', 'ro', 'mo', 'do', 'so']) and not col.endswith('_type')],
            'metadata_fields': [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume', 'range']]
        }
        
        for group_name, fields in field_groups.items():
            if fields:
                print(f"   {group_name}: {len(fields)} полей")
                
                # Показываем примеры значений для ключевых полей
                if group_name in ['nw_fields', 'ef_fields', 'as_fields', 'vc_fields', 'ze_fields']:
                    for field in fields[:3]:  # Первые 3 поля
                        non_null = df[field].dropna()
                        if len(non_null) > 0:
                            print(f"      {field}: мин={non_null.min():.2f}, макс={non_null.max():.2f}, активаций={len(non_null)}")
        
        # Проверка критических полей
        critical_fields = ['nw2', 'ef2', 'as2', 'vc2', 'ze2']
        print(f"\n🎯 КРИТИЧЕСКИЕ ПОЛЯ:")
        for field in critical_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                if non_null_count > 0:
                    sample_values = df[field].dropna().head(3).tolist()
                    print(f"   ✅ {field}: найдено {non_null_count} активаций, примеры: {sample_values}")
                else:
                    print(f"   ⚠️ {field}: поле есть, но все значения NULL")
            else:
                print(f"   ❌ {field}: НЕ НАЙДЕНО")
    
    def get_ltf_htf_separation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        УНИВЕРСАЛЬНОЕ разделение полей на LTF и HTF по типам
        
        Returns:
            (ltf_df, htf_df): отдельные датафреймы для LTF и HTF
        """
        print("⚡ Разделение на LTF/HTF по типам полей...")
        
        # Общие поля (метаданные свечей)
        common_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'range', 
                        'candle_color', 'candle_type', 'line_number', 'raw_line']
        
        ltf_columns = [col for col in common_fields if col in df.columns]
        htf_columns = [col for col in common_fields if col in df.columns]
        
        # Распределение полей по типам
        for col in df.columns:
            if col.endswith('_type'):
                continue  # Пропускаем служебные поля типов
                
            # Проверяем наличие типа поля
            type_col = f"{col}_type"
            if type_col in df.columns:
                field_types = df[type_col].dropna().unique()
                
                # LTF поля
                if any('LTF' in str(t) for t in field_types):
                    ltf_columns.append(col)
                    if type_col in df.columns:
                        ltf_columns.append(type_col)
                
                # HTF поля  
                if any('HTF' in str(t) for t in field_types):
                    htf_columns.append(col)
                    if type_col in df.columns:
                        htf_columns.append(type_col)
            else:
                # Fallback: определяем по суффиксам
                for suffix in self.ltf_suffixes:
                    if col.endswith(suffix):
                        ltf_columns.append(col)
                        break
                else:
                    for suffix in self.htf_suffixes:
                        if col.endswith(suffix):
                            htf_columns.append(col)
                            break
        
        # Создание отдельных датафреймов
        ltf_df = df[[col for col in set(ltf_columns) if col in df.columns]].copy()
        htf_df = df[[col for col in set(htf_columns) if col in df.columns]].copy()
        
        # Статистика
        ltf_indicator_fields = [col for col in ltf_df.columns 
                               if not col in common_fields and not col.endswith('_type')]
        htf_indicator_fields = [col for col in htf_df.columns 
                               if not col in common_fields and not col.endswith('_type')]
        
        print(f"   LTF: {len(ltf_indicator_fields)} индикаторных полей")
        print(f"   HTF: {len(htf_indicator_fields)} индикаторных полей")
        
        # Показываем примеры полей
        if ltf_indicator_fields:
            print(f"   LTF примеры: {', '.join(ltf_indicator_fields[:5])}")
        if htf_indicator_fields:
            print(f"   HTF примеры: {', '.join(htf_indicator_fields[:5])}")
        
        return ltf_df, htf_df
    
    def validate_parsing_quality(self, df: pd.DataFrame) -> Dict:
        """Валидация качества парсинга"""
        print("🔍 Валидация качества парсинга...")
        
        validation_results = {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'critical_fields_found': 0,
            'parsing_quality_score': 0.0,
            'field_coverage': {}
        }
        
        # Проверка критических полей
        critical_fields = ['nw2', 'ef2', 'as2', 'vc2', 'ze2']
        for field in critical_fields:
            if field in df.columns and df[field].notna().sum() > 0:
                validation_results['critical_fields_found'] += 1
        
        # Оценка качества парсинга
        parsing_quality = validation_results['critical_fields_found'] / len(critical_fields)
        validation_results['parsing_quality_score'] = parsing_quality
        
        if parsing_quality >= 0.8:
            print("   ✅ Отличное качество парсинга!")
        elif parsing_quality >= 0.6:
            print("   ⚠️ Хорошее качество парсинга")
        else:
            print("   ❌ Низкое качество парсинга - проверьте форматы")
        
        return validation_results


# Функция для быстрого тестирования
def test_parser_on_sample():
    """Тестирование ИСПРАВЛЕННОГО парсера на примере данных"""
    sample_line = """[2024-08-05T09:24:00.000+03:00]: LTF|event_2025-06-28_22-55|1|2024-08-05 06:24|RED|-1.79%|11.5K|BIG_BODY|66%|-18.8%_24h|o:50254.8|h:50258.6|l:48888|c:49353.4|rng:1370.6|p2-0,p5-80,p15-60,p30-80,md5-47%,md15-206.5%,md30-153.3%,cmd5-1.1%,cmd30-11%,macd5-9.1%,ro2-11,ro5-15,ro15-19,ro30-12,mo2-12,mo5-15,mo30-15,co2--213,co5--299,co15--316,co30--153,cz2--0.24,cz5--0.31,cz15--0.2,cz30--0.23,do5-32,do15-31,do30-32,so2-11,so5-9,so15-8,so30-5,rz2--2.63,rz5--2.44,rz15--1.53,rz30--2.3,mz2--2.29,mz30--1.7,ciz2--1.66,ciz5--2.42,ciz15--2.17,sz30--1.58,dz5--2.21,dz15--1.89,dz30--1.83,cvz2--2.55,cvz5--1.71,cvz15--2.36,cvz30--3.41,maz2--4.06,maz15--2.23,maz30--3.7,ef2--7.19,ef5--4.32,ef15--3.72,ef30--5.03,vc2-2.4,vc5-3.3,vc15-3,ze2--4.12,ze5--2.5,ze15--2.71,ze30--4.6,nw2-!!,nw5-!!,nw15-!!,nw30-!!,as2-3.33,as5-4.39,as15-3.56,as30-4.31,vw2--3.09,vw5--3,vw15--2.47"""
    
    parser = AdvancedLogParser()
    record = parser._parse_single_line(sample_line, 0)
    
    print("🧪 ТЕСТ ИСПРАВЛЕННОГО ПАРСЕРА:")
    print(f"Извлечено полей: {len(record)}")
    print()
    
    # Проверка критических полей
    critical_fields = [
        ('nw2', '!!'), ('ef2', -7.19), ('as2', 3.33), ('vc2', 2.4), ('ze2', -4.12)
    ]
    
    print("🎯 ПРОВЕРКА КРИТИЧЕСКИХ ПОЛЕЙ:")
    for field_name, expected_value in critical_fields:
        if field_name in record:
            actual_value = record[field_name]
            if actual_value == expected_value:
                print(f"   ✅ {field_name}: {actual_value} (ПРАВИЛЬНО)")
            else:
                print(f"   ⚠️ {field_name}: {actual_value} (ожидалось {expected_value})")
        else:
            print(f"   ❌ {field_name}: НЕ НАЙДЕНО")
    
    # Показываем все ef и ze поля
    print(f"\n🔥 ВСЕ EF ПОЛЯ:")
    for key, value in record.items():
        if key.startswith('ef') and not key.endswith('_type'):
            print(f"   {key}: {value}")
    
    print(f"\n🔥 ВСЕ ZE ПОЛЯ:")
    for key, value in record.items():
        if key.startswith('ze') and not key.endswith('_type'):
            print(f"   {key}: {value}")


if __name__ == "__main__":
    print("🚀 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО ПАРСЕРА")
    print("=" * 50)
    
    test_parser_on_sample()
    
    print("\n🎯 ГОТОВ К РАБОТЕ С РЕАЛЬНЫМИ ДАННЫМИ!")
