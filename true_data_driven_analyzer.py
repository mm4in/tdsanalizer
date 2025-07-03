#!/usr/bin/env python3
"""
ПРАВИЛЬНЫЙ DATA-DRIVEN АНАЛИЗАТОР - БЕЗ ПРЕДВЗЯТОСТИ!

Цель: Найти паттерны для дамп/памп событий на основе СТАТИСТИКИ, а не предположений!

Типы событий:
1. ЛОИ с откатами ВВЕРХ 3%+ (дамп → контртренд покупка)
2. ХАИ с откатами ВНИЗ 3%+ (памп → контртренд продажа)  
3. Продолжения дампа (без откатов)
4. Продолжения пампа (без откатов)

DATA-DRIVEN: Анализируем ВСЕ поля без априорных предположений!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re
import json
from pathlib import Path
from datetime import datetime

class TrueDataDrivenAnalyzer:
    """ПРАВИЛЬНЫЙ анализатор без предвзятости"""
    
    def __init__(self):
        self.all_data = []
        self.all_fields_stats = {}
        self.events = []
        self.field_correlations = {}
        
    def parse_line_all_fields(self, line: str) -> Dict:
        """Извлечение ВСЕХ полей без предвзятости"""
        fields = {}
        
        # Метаданные свечей
        ohlc_match = re.search(r'o:([0-9.]+)\|h:([0-9.]+)\|l:([0-9.]+)\|c:([0-9.]+)', line)
        if ohlc_match:
            fields['open'] = float(ohlc_match.group(1))
            fields['high'] = float(ohlc_match.group(2))
            fields['low'] = float(ohlc_match.group(3))
            fields['close'] = float(ohlc_match.group(4))
        
        volume_match = re.search(r'\|([0-9.]+)K\|', line)
        if volume_match:
            fields['volume'] = float(volume_match.group(1))
        
        range_match = re.search(r'rng:([0-9.]+)', line)
        if range_match:
            fields['range'] = float(range_match.group(1))
        
        color_match = re.search(r'\|(RED|GREEN)\|', line)
        if color_match:
            fields['candle_color'] = color_match.group(1)
        
        change_match = re.search(r'\|(RED|GREEN)\|(-?[0-9.]+)%\|', line)
        if change_match:
            fields['price_change_pct'] = float(change_match.group(2))
        
        # Универсальный паттерн для ВСЕХ полей данных
        universal_pattern = r'([a-zA-Z]+)(\d+|1h|4h|1d|1w)-((?:!+)|(?:--?\d+(?:\.\d+)?(?:%)?)|(?:-?\d+(?:\.\d+)?(?:%)?))'
        
        for match in re.finditer(universal_pattern, line):
            prefix = match.group(1)
            suffix = match.group(2)
            value = match.group(3)
            field_name = f"{prefix}{suffix}"
            
            # Обработка значений
            if '!' in value:
                # Сигналы типа !!, !!!
                fields[field_name] = len(value)
                fields[f"{field_name}_signal"] = value
            elif '%' in value:
                # Процентные значения
                num_value = float(value.replace('%', ''))
                if value.startswith('--'):
                    fields[field_name] = -num_value
                else:
                    fields[field_name] = num_value
            else:
                # Числовые значения
                if value.startswith('--'):
                    fields[field_name] = -float(value[2:])
                elif value.startswith('-'):
                    fields[field_name] = -float(value[1:])
                else:
                    fields[field_name] = float(value)
        
        return fields
    
    def load_and_parse_data(self, file_path: str) -> None:
        """Загрузка и парсинг ВСЕХ данных"""
        print(f"🔄 Загрузка данных: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📋 Найдено {len(lines)} строк")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                fields = self.parse_line_all_fields(line)
                if fields and 'close' in fields:
                    fields['line_number'] = i
                    fields['timestamp'] = self._extract_timestamp(line)
                    self.all_data.append(fields)
                    
                if (i + 1) % 1000 == 0:
                    print(f"   Обработано: {i + 1}/{len(lines)} строк")
                    
            except Exception as e:
                print(f"⚠️ Ошибка в строке {i}: {str(e)[:100]}")
                continue
        
        print(f"✅ Загружено {len(self.all_data)} записей")
        
        # Создаем DataFrame
        self.df = pd.DataFrame(self.all_data)
        self.df = self.df.sort_values('line_number').reset_index(drop=True)
        
        print(f"📊 Создан DataFrame: {len(self.df)} строк × {len(self.df.columns)} столбцов")
        
        # Показываем статистику по полям
        self._analyze_field_coverage()
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Извлечение timestamp"""
        ts_match = re.search(r'\[([^\]]+)\]', line)
        return ts_match.group(1) if ts_match else None
    
    def _analyze_field_coverage(self) -> None:
        """Анализ покрытия полей"""
        print("\n📊 АНАЛИЗ ПОКРЫТИЯ ПОЛЕЙ:")
        
        # Исключаем служебные поля
        exclude_fields = {'line_number', 'timestamp', 'open', 'high', 'low', 'close', 
                         'volume', 'range', 'candle_color', 'price_change_pct'}
        
        indicator_fields = [col for col in self.df.columns 
                           if col not in exclude_fields and not col.endswith('_signal')]
        
        print(f"   Всего индикаторных полей: {len(indicator_fields)}")
        
        # Группируем по префиксам
        field_groups = defaultdict(list)
        for field in indicator_fields:
            prefix = re.match(r'^([a-zA-Z]+)', field)
            if prefix:
                field_groups[prefix.group(1)].append(field)
        
        print(f"   Групп полей: {len(field_groups)}")
        for prefix, fields in sorted(field_groups.items()):
            coverage = sum(self.df[field].notna().sum() for field in fields)
            print(f"     {prefix}: {len(fields)} полей, {coverage} активаций")
    
    def detect_extrema_events(self) -> List[Dict]:
        """Поиск экстремумов и определение типов событий"""
        print("\n🎯 ПОИСК ЭКСТРЕМУМОВ И СОБЫТИЙ...")
        
        if len(self.df) < 50:
            print("❌ Недостаточно данных для анализа")
            return []
        
        events = []
        
        # Скользящие окна для поиска локальных экстремумов
        window = 10
        
        # Находим локальные минимумы
        for i in range(window, len(self.df) - window - 30):  # Оставляем 30 записей для анализа откатов
            current_low = self.df.iloc[i]['low']
            
            # Проверяем что это локальный минимум
            is_local_min = True
            for j in range(i - window, i + window + 1):
                if j != i and self.df.iloc[j]['low'] <= current_low:
                    is_local_min = False
                    break
            
            if is_local_min:
                event = self._analyze_low_event(i)
                if event:
                    events.append(event)
        
        # Находим локальные максимумы
        for i in range(window, len(self.df) - window - 30):
            current_high = self.df.iloc[i]['high']
            
            # Проверяем что это локальный максимум
            is_local_max = True
            for j in range(i - window, i + window + 1):
                if j != i and self.df.iloc[j]['high'] >= current_high:
                    is_local_max = False
                    break
            
            if is_local_max:
                event = self._analyze_high_event(i)
                if event:
                    events.append(event)
        
        self.events = events
        print(f"✅ Найдено {len(events)} событий")
        
        # Статистика по типам
        event_types = defaultdict(int)
        for event in events:
            event_types[event['type']] += 1
        
        print("📈 Типы событий:")
        for event_type, count in sorted(event_types.items()):
            print(f"   {event_type}: {count}")
        
        return events
    
    def _analyze_low_event(self, idx: int) -> Optional[Dict]:
        """Анализ события лои"""
        low_price = self.df.iloc[idx]['low']
        
        # Смотрим следующие 30 записей для поиска отката
        future_data = self.df.iloc[idx:idx+30]
        if len(future_data) < 10:
            return None
        
        max_high_after = future_data['high'].max()
        rebound_pct = ((max_high_after - low_price) / low_price) * 100
        
        # Определяем тип события
        if rebound_pct >= 3.0:
            event_type = "low_with_rebound_3pct"
        elif rebound_pct >= 2.0:
            event_type = "low_with_rebound_2pct"
        elif rebound_pct >= 1.0:
            event_type = "low_with_rebound_1pct"
        else:
            event_type = "low_no_rebound"
        
        # Извлекаем ВСЕ индикаторы на момент события
        indicators = self._extract_all_indicators_at_moment(idx)
        
        return {
            'type': event_type,
            'category': 'low_event',
            'index': idx,
            'timestamp': self.df.iloc[idx].get('timestamp', ''),
            'price': low_price,
            'rebound_pct': rebound_pct,
            'max_price_after': max_high_after,
            'indicators': indicators
        }
    
    def _analyze_high_event(self, idx: int) -> Optional[Dict]:
        """Анализ события хаи"""
        high_price = self.df.iloc[idx]['high']
        
        # Смотрим следующие 30 записей для поиска отката
        future_data = self.df.iloc[idx:idx+30]
        if len(future_data) < 10:
            return None
        
        min_low_after = future_data['low'].min()
        decline_pct = ((high_price - min_low_after) / high_price) * 100
        
        # Определяем тип события
        if decline_pct >= 3.0:
            event_type = "high_with_decline_3pct"
        elif decline_pct >= 2.0:
            event_type = "high_with_decline_2pct"
        elif decline_pct >= 1.0:
            event_type = "high_with_decline_1pct"
        else:
            event_type = "high_no_decline"
        
        # Извлекаем ВСЕ индикаторы на момент события
        indicators = self._extract_all_indicators_at_moment(idx)
        
        return {
            'type': event_type,
            'category': 'high_event',
            'index': idx,
            'timestamp': self.df.iloc[idx].get('timestamp', ''),
            'price': high_price,
            'decline_pct': decline_pct,
            'min_price_after': min_low_after,
            'indicators': indicators
        }
    
    def _extract_all_indicators_at_moment(self, idx: int) -> Dict:
        """Извлечение ВСЕХ индикаторов на момент события"""
        row = self.df.iloc[idx]
        indicators = {}
        
        # Исключаем метаданные
        exclude_fields = {'line_number', 'timestamp', 'open', 'high', 'low', 'close', 
                         'volume', 'range', 'candle_color', 'price_change_pct'}
        
        for field in self.df.columns:
            if field not in exclude_fields and not field.endswith('_signal'):
                value = row[field]
                if pd.notna(value):
                    indicators[field] = value
        
        return indicators
    
    def analyze_field_correlations(self) -> Dict:
        """DATA-DRIVEN анализ корреляций полей с типами событий"""
        print("\n🧠 DATA-DRIVEN АНАЛИЗ КОРРЕЛЯЦИЙ...")
        
        if not self.events:
            print("❌ Сначала найдите события")
            return {}
        
        correlations = {}
        
        # Группируем события по типам
        events_by_type = defaultdict(list)
        for event in self.events:
            events_by_type[event['type']].append(event)
        
        print(f"Анализируем {len(events_by_type)} типов событий...")
        
        # Анализ каждого типа события
        for event_type, events_list in events_by_type.items():
            if len(events_list) < 3:  # Минимум для статистики
                continue
            
            print(f"\n📊 Анализ: {event_type} ({len(events_list)} событий)")
            
            # Собираем статистику по всем полям
            field_stats = self._calculate_field_statistics_for_events(events_list)
            correlations[event_type] = field_stats
        
        self.field_correlations = correlations
        return correlations
    
    def _calculate_field_statistics_for_events(self, events_list: List[Dict]) -> Dict:
        """Расчет статистики полей для списка событий"""
        all_fields = set()
        for event in events_list:
            all_fields.update(event['indicators'].keys())
        
        field_stats = {}
        
        for field in all_fields:
            values = []
            for event in events_list:
                if field in event['indicators']:
                    value = event['indicators'][field]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        values.append(value)
            
            if len(values) >= 2:  # Минимум для статистики
                field_stats[field] = {
                    'activation_count': len(values),
                    'activation_rate': len(values) / len(events_list),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'abs_mean': np.mean(np.abs(values))  # Для ранжирования
                }
        
        return field_stats
    
    def find_discriminative_fields(self) -> Dict:
        """Поиск полей различающих типы событий"""
        print("\n🔍 ПОИСК ДИСКРИМИНАТИВНЫХ ПОЛЕЙ...")
        
        if not self.field_correlations:
            print("❌ Сначала выполните анализ корреляций")
            return {}
        
        discriminative = {}
        
        # Сравниваем основные пары типов событий
        comparison_pairs = [
            ("low_with_rebound_3pct", "low_no_rebound"),
            ("high_with_decline_3pct", "high_no_decline"),
            ("low_with_rebound_3pct", "high_with_decline_3pct")
        ]
        
        for type1, type2 in comparison_pairs:
            if type1 in self.field_correlations and type2 in self.field_correlations:
                diff = self._compare_field_patterns(type1, type2)
                if diff:
                    discriminative[f"{type1}_vs_{type2}"] = diff
        
        return discriminative
    
    def _compare_field_patterns(self, type1: str, type2: str) -> Dict:
        """Сравнение паттернов полей между типами событий"""
        stats1 = self.field_correlations[type1]
        stats2 = self.field_correlations[type2]
        
        differences = {}
        common_fields = set(stats1.keys()) & set(stats2.keys())
        
        for field in common_fields:
            s1 = stats1[field]
            s2 = stats2[field]
            
            # Сила различия
            mean_diff = abs(s1['mean'] - s2['mean'])
            activation_diff = abs(s1['activation_rate'] - s2['activation_rate'])
            
            # Показатель дискриминативности
            discriminative_power = mean_diff + activation_diff * 10
            
            if discriminative_power > 0.5:
                differences[field] = {
                    'discriminative_power': discriminative_power,
                    'mean_diff': mean_diff,
                    'activation_diff': activation_diff,
                    f'{type1}_mean': s1['mean'],
                    f'{type2}_mean': s2['mean'],
                    f'{type1}_activation': s1['activation_rate'],
                    f'{type2}_activation': s2['activation_rate']
                }
        
        # Сортируем по дискриминативной силе
        return dict(sorted(differences.items(), 
                          key=lambda x: x[1]['discriminative_power'], 
                          reverse=True))
    
    def generate_data_driven_tables(self) -> Dict:
        """Создание таблиц на основе данных"""
        print("\n📋 СОЗДАНИЕ DATA-DRIVEN ТАБЛИЦ...")
        
        if not self.field_correlations:
            return {}
        
        tables = {}
        
        for event_type, field_stats in self.field_correlations.items():
            # Ранжируем поля по важности
            ranked_fields = []
            for field, stats in field_stats.items():
                importance = stats['activation_rate'] * (1 + stats['abs_mean'])
                ranked_fields.append({
                    'field': field,
                    'importance': importance,
                    'activation_rate': stats['activation_rate'],
                    'activation_count': stats['activation_count'],
                    'mean': stats['mean'],
                    'abs_mean': stats['abs_mean']
                })
            
            # Сортируем по важности
            ranked_fields.sort(key=lambda x: x['importance'], reverse=True)
            tables[event_type] = ranked_fields[:30]  # Топ 30
        
        return tables
    
    def save_results(self, output_dir: str = "results/true_data_driven"):
        """Сохранение результатов"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Сохранение результатов в {output_path}")
        
        # События
        if self.events:
            events_df = pd.DataFrame([
                {
                    'type': event['type'],
                    'category': event['category'],
                    'timestamp': event['timestamp'],
                    'price': event['price'],
                    'change_pct': event.get('rebound_pct', event.get('decline_pct', 0)),
                    'indicator_count': len(event['indicators'])
                }
                for event in self.events
            ])
            events_df.to_csv(output_path / "events_summary.csv", index=False)
        
        # Корреляции полей
        if self.field_correlations:
            with open(output_path / "field_correlations.json", 'w', encoding='utf-8') as f:
                json.dump(self.field_correlations, f, indent=2, ensure_ascii=False, default=str)
        
        # Data-driven таблицы
        tables = self.generate_data_driven_tables()
        if tables:
            with open(output_path / "data_driven_tables.json", 'w', encoding='utf-8') as f:
                json.dump(tables, f, indent=2, ensure_ascii=False, default=str)
        
        # Дискриминативные поля
        discriminative = self.find_discriminative_fields()
        if discriminative:
            with open(output_path / "discriminative_fields.json", 'w', encoding='utf-8') as f:
                json.dump(discriminative, f, indent=2, ensure_ascii=False, default=str)
        
        # Понятный отчет
        self._create_readable_report(output_path, tables, discriminative)
        
        return output_path
    
    def _create_readable_report(self, output_path: Path, tables: Dict, discriminative: Dict):
        """Создание понятного отчета"""
        report_lines = [
            "# ПРАВИЛЬНЫЙ DATA-DRIVEN АНАЛИЗ ДАМП/ПАМП",
            "=" * 60,
            "",
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Всего записей: {len(self.df)}",
            f"Найдено событий: {len(self.events)}",
            f"Проанализировано полей: {len(self.df.columns)}",
            "",
            "## СТАТИСТИКА СОБЫТИЙ",
            ""
        ]
        
        # Статистика по типам
        if self.events:
            event_counts = defaultdict(int)
            for event in self.events:
                event_counts[event['type']] += 1
            
            for event_type, count in sorted(event_counts.items()):
                report_lines.append(f"{event_type}: {count} событий")
        
        # Топ поля по типам событий
        if tables:
            for event_type, fields in tables.items():
                report_lines.extend([
                    "",
                    f"## ТОП ПОЛЯ ДЛЯ {event_type.upper()}",
                    f"(на основе статистического анализа {len([e for e in self.events if e['type'] == event_type])} событий)",
                    ""
                ])
                
                for i, field_data in enumerate(fields[:15], 1):
                    field = field_data['field']
                    importance = field_data['importance']
                    activation = field_data['activation_rate']
                    mean = field_data['mean']
                    count = field_data['activation_count']
                    
                    report_lines.append(
                        f"{i:2d}. {field:20s} | "
                        f"важность: {importance:8.2f} | "
                        f"активация: {activation:5.1%} | "
                        f"среднее: {mean:8.2f} | "
                        f"событий: {count}"
                    )
        
        # Дискриминативные поля
        if discriminative:
            report_lines.extend([
                "",
                "## ПОЛЯ РАЗЛИЧАЮЩИЕ ТИПЫ СОБЫТИЙ",
                "(статистически значимые различия)",
                ""
            ])
            
            for comparison, fields in discriminative.items():
                report_lines.append(f"### {comparison}")
                
                for field, data in list(fields.items())[:10]:
                    power = data['discriminative_power']
                    report_lines.append(
                        f"  {field}: дискриминативная сила = {power:.2f}"
                    )
                report_lines.append("")
        
        # Сохраняем отчет
        with open(output_path / "data_driven_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print("   ✅ Создан data-driven отчет")


def main():
    """Основная функция"""
    print("🚀 ПРАВИЛЬНЫЙ DATA-DRIVEN АНАЛИЗ ДАМП/ПАМП")
    print("=" * 80)
    print("БЕЗ ПРЕДВЗЯТОСТИ! ТОЛЬКО СТАТИСТИКА!")
    print("=" * 80)
    
    analyzer = TrueDataDrivenAnalyzer()
    
    try:
        # 1. Загружаем данные
        print("\n1️⃣ ЗАГРУЗКА ДАННЫХ")
        analyzer.load_and_parse_data("data/dslog_btc_0508240229_ltf.txt")
        
        # 2. Находим события
        print("\n2️⃣ ПОИСК СОБЫТИЙ")
        events = analyzer.detect_extrema_events()
        
        if not events:
            print("❌ События не найдены")
            return
        
        # 3. Анализ корреляций
        print("\n3️⃣ DATA-DRIVEN АНАЛИЗ КОРРЕЛЯЦИЙ")
        correlations = analyzer.analyze_field_correlations()
        
        # 4. Дискриминативный анализ
        print("\n4️⃣ ПОИСК ДИСКРИМИНАТИВНЫХ ПОЛЕЙ")
        discriminative = analyzer.find_discriminative_fields()
        
        # 5. Сохранение
        print("\n5️⃣ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        output_path = analyzer.save_results()
        
        print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📁 Результаты: {output_path}")
        print(f"📋 Основной отчет: {output_path}/data_driven_report.txt")
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
