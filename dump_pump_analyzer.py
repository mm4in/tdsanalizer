#!/usr/bin/env python3
"""
АНАЛИЗАТОР ПАТТЕРНОВ ДЛЯ КОНТРТРЕНДОВОГО СКАЛЬПИНГА
Задача: Найти паттерны для дамп/памп сценариев

ТИПЫ СОБЫТИЙ:
1. ЛОИ с откатами ВВЕРХ 3%+ (дамп → контртренд покупка)
2. ХАИ с откатами ВНИЗ 3%+ (памп → контртренд продажа)  
3. Продолжения дампа (без откатов, сильное падение)
4. Продолжения пампа (без откатов, сильный рост)

DATA-DRIVEN ПОДХОД: анализируем ВСЕ поля без предвзятости!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from advanced_log_parser import AdvancedLogParser
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DumpPumpAnalyzer:
    """Анализатор паттернов для контртрендового скальпинга"""
    
    def __init__(self):
        self.parser = AdvancedLogParser()
        self.data = None
        self.events = []
        self.patterns = {}
        
    def load_and_parse_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка и парсинг данных"""
        print("🔄 Загрузка и парсинг данных...")
        
        # Парсинг данных
        self.data = self.parser.parse_log_file(file_path)
        
        if self.data.empty:
            raise ValueError("❌ Не удалось загрузить данные")
            
        # Предобработка данных
        self.data = self._preprocess_data(self.data)
        
        print(f"✅ Загружено {len(self.data)} записей с {len(self.data.columns)} полями")
        return self.data
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных"""
        print("🔧 Предобработка данных...")
        
        # Сортировка по времени
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Расчет процентного изменения цены
        if 'close' in df.columns:
            df['price_change_pct'] = df['close'].pct_change() * 100
            df['price_change_abs'] = df['close'].diff()
        
        # Определение трендов
        if 'close' in df.columns:
            df['trend_5'] = df['close'].rolling(5).mean().diff() > 0
            df['trend_20'] = df['close'].rolling(20).mean().diff() > 0
        
        # Вычисление rolling минимумов и максимумов для поиска экстремумов
        if 'low' in df.columns and 'high' in df.columns:
            df['rolling_low_5'] = df['low'].rolling(5, center=True).min()
            df['rolling_high_5'] = df['high'].rolling(5, center=True).max()
            df['rolling_low_10'] = df['low'].rolling(10, center=True).min()
            df['rolling_high_10'] = df['high'].rolling(10, center=True).max()
            
            # Локальные экстремумы
            df['is_local_low'] = (df['low'] == df['rolling_low_5']) & (df['low'] == df['rolling_low_10'])
            df['is_local_high'] = (df['high'] == df['rolling_high_5']) & (df['high'] == df['rolling_high_10'])
        
        return df
    
    def detect_events(self) -> List[Dict]:
        """Обнаружение событий: лои/хаи с откатами и продолжения"""
        print("🎯 Поиск событий...")
        
        if self.data is None or self.data.empty:
            print("❌ Нет данных для анализа")
            return []
        
        events = []
        
        # Находим локальные минимумы и максимумы
        lows = self.data[self.data['is_local_low'] == True].copy()
        highs = self.data[self.data['is_local_high'] == True].copy()
        
        print(f"Найдено локальных минимумов: {len(lows)}")
        print(f"Найдено локальных максимумов: {len(highs)}")
        
        # Анализ лои (потенциальные дампы)
        for idx, low_row in lows.iterrows():
            event = self._analyze_low_event(idx, low_row)
            if event:
                events.append(event)
        
        # Анализ хаи (потенциальные пампы)
        for idx, high_row in highs.iterrows():
            event = self._analyze_high_event(idx, high_row)
            if event:
                events.append(event)
        
        self.events = events
        print(f"✅ Найдено {len(events)} событий")
        
        # Статистика по типам событий
        event_types = {}
        for event in events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("📊 Статистика событий:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")
        
        return events
    
    def _analyze_low_event(self, idx: int, low_row: pd.Series) -> Optional[Dict]:
        """Анализ лои: дамп с откатом или продолжение падения"""
        
        # Ищем следующие 30 записей после лои
        next_data = self.data.iloc[idx:idx+30].copy() if idx+30 < len(self.data) else self.data.iloc[idx:].copy()
        
        if len(next_data) < 5:
            return None
        
        low_price = low_row['low']
        
        # Ищем максимальный отскок после лои
        max_high_after = next_data['high'].max()
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
        
        # Извлекаем индикаторы на момент лои
        indicators = self._extract_indicators_at_moment(idx)
        
        return {
            'type': event_type,
            'category': 'low_event',
            'index': idx,
            'timestamp': low_row.get('timestamp', ''),
            'price': low_price,
            'rebound_pct': rebound_pct,
            'max_price_after': max_high_after,
            'indicators': indicators
        }
    
    def _analyze_high_event(self, idx: int, high_row: pd.Series) -> Optional[Dict]:
        """Анализ хаи: памп с откатом или продолжение роста"""
        
        # Ищем следующие 30 записей после хаи
        next_data = self.data.iloc[idx:idx+30].copy() if idx+30 < len(self.data) else self.data.iloc[idx:].copy()
        
        if len(next_data) < 5:
            return None
        
        high_price = high_row['high']
        
        # Ищем минимальный откат после хаи
        min_low_after = next_data['low'].min()
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
        
        # Извлекаем индикаторы на момент хаи
        indicators = self._extract_indicators_at_moment(idx)
        
        return {
            'type': event_type,
            'category': 'high_event',
            'index': idx,
            'timestamp': high_row.get('timestamp', ''),
            'price': high_price,
            'decline_pct': decline_pct,
            'min_price_after': min_low_after,
            'indicators': indicators
        }
    
    def _extract_indicators_at_moment(self, idx: int) -> Dict:
        """Извлечение ВСЕХ индикаторов на момент события"""
        if idx >= len(self.data):
            return {}
        
        row = self.data.iloc[idx]
        indicators = {}
        
        # Извлекаем ВСЕ поля кроме метаданных
        exclude_fields = {
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'range',
            'candle_color', 'candle_type', 'line_number', 'raw_line',
            'price_change_pct', 'price_change_abs', 'trend_5', 'trend_20',
            'rolling_low_5', 'rolling_high_5', 'rolling_low_10', 'rolling_high_10',
            'is_local_low', 'is_local_high'
        }
        
        for field in self.data.columns:
            if field not in exclude_fields and not field.endswith('_type'):
                value = row[field]
                if pd.notna(value):  # Только не-NaN значения
                    indicators[field] = value
        
        return indicators
    
    def analyze_patterns(self) -> Dict:
        """DATA-DRIVEN анализ паттернов для каждого типа события"""
        print("🧠 DATA-DRIVEN анализ паттернов...")
        
        if not self.events:
            print("❌ Нет событий для анализа")
            return {}
        
        patterns = {}
        
        # Группируем события по типам
        events_by_type = {}
        for event in self.events:
            event_type = event['type']
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        print(f"Анализируем {len(events_by_type)} типов событий...")
        
        # Анализ каждого типа события
        for event_type, events_list in events_by_type.items():
            if len(events_list) < 3:  # Минимум 3 события для анализа
                continue
                
            print(f"\n📈 Анализ типа: {event_type} ({len(events_list)} событий)")
            
            # Собираем все индикаторы для этого типа
            all_indicators = {}
            for event in events_list:
                for indicator, value in event['indicators'].items():
                    if indicator not in all_indicators:
                        all_indicators[indicator] = []
                    all_indicators[indicator].append(value)
            
            # Статистический анализ каждого индикатора
            indicator_stats = {}
            for indicator, values in all_indicators.items():
                if len(values) >= 2:  # Минимум 2 значения
                    stats = self._calculate_indicator_statistics(indicator, values)
                    indicator_stats[indicator] = stats
            
            patterns[event_type] = {
                'count': len(events_list),
                'indicator_stats': indicator_stats,
                'events': events_list
            }
        
        self.patterns = patterns
        return patterns
    
    def _calculate_indicator_statistics(self, indicator: str, values: List) -> Dict:
        """Расчет статистики для индикатора"""
        # Фильтруем числовые значения
        numeric_values = []
        string_values = []
        
        for value in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(value)
            else:
                string_values.append(str(value))
        
        stats = {
            'total_activations': len(values),
            'numeric_count': len(numeric_values),
            'string_count': len(string_values)
        }
        
        # Статистика для числовых значений
        if numeric_values:
            numeric_values = np.array(numeric_values)
            stats.update({
                'mean': float(np.mean(numeric_values)),
                'median': float(np.median(numeric_values)),
                'std': float(np.std(numeric_values)),
                'min': float(np.min(numeric_values)),
                'max': float(np.max(numeric_values)),
                'activation_rate': len(numeric_values) / len(values)
            })
        
        # Статистика для строковых значений
        if string_values:
            from collections import Counter
            value_counts = Counter(string_values)
            stats['string_patterns'] = dict(value_counts)
        
        return stats
    
    def find_discriminative_patterns(self) -> Dict:
        """Поиск дискриминативных паттернов между типами событий"""
        print("🔍 Поиск дискриминативных паттернов...")
        
        if not self.patterns:
            print("❌ Сначала запустите analyze_patterns()")
            return {}
        
        discriminative = {}
        
        # Сравниваем основные типы событий
        comparison_pairs = [
            ("low_with_rebound_3pct", "low_no_rebound"),
            ("high_with_decline_3pct", "high_no_decline"),
            ("low_with_rebound_3pct", "high_with_decline_3pct")
        ]
        
        for type1, type2 in comparison_pairs:
            if type1 in self.patterns and type2 in self.patterns:
                diff = self._compare_pattern_types(type1, type2)
                discriminative[f"{type1}_vs_{type2}"] = diff
        
        return discriminative
    
    def _compare_pattern_types(self, type1: str, type2: str) -> Dict:
        """Сравнение двух типов паттернов"""
        pattern1 = self.patterns[type1]['indicator_stats']
        pattern2 = self.patterns[type2]['indicator_stats']
        
        differences = {}
        
        # Общие индикаторы
        common_indicators = set(pattern1.keys()) & set(pattern2.keys())
        
        for indicator in common_indicators:
            stats1 = pattern1[indicator]
            stats2 = pattern2[indicator]
            
            # Сравниваем только если есть числовые значения
            if 'mean' in stats1 and 'mean' in stats2:
                mean_diff = abs(stats1['mean'] - stats2['mean'])
                activation_diff = abs(stats1.get('activation_rate', 0) - stats2.get('activation_rate', 0))
                
                # Показатель различимости
                discriminative_power = mean_diff + activation_diff * 10
                
                if discriminative_power > 0.5:  # Пороговое значение
                    differences[indicator] = {
                        'mean_diff': mean_diff,
                        'activation_diff': activation_diff,
                        'discriminative_power': discriminative_power,
                        f'{type1}_mean': stats1['mean'],
                        f'{type2}_mean': stats2['mean'],
                        f'{type1}_activation': stats1.get('activation_rate', 0),
                        f'{type2}_activation': stats2.get('activation_rate', 0)
                    }
        
        # Сортируем по дискриминативной силе
        sorted_differences = dict(sorted(differences.items(), 
                                       key=lambda x: x[1]['discriminative_power'], 
                                       reverse=True))
        
        return sorted_differences
    
    def find_veto_patterns(self) -> Dict:
        """Поиск VETO паттернов - полей блокирующих ложные сигналы"""
        print("🚫 Поиск VETO паттернов...")
        
        if not self.patterns:
            return {}
        
        veto_patterns = {}
        
        # Ищем поля которые активны в "плохих" сценариях и неактивны в "хороших"
        good_events = ['low_with_rebound_3pct', 'high_with_decline_3pct']
        bad_events = ['low_no_rebound', 'high_no_decline']
        
        for good_type in good_events:
            for bad_type in bad_events:
                if good_type in self.patterns and bad_type in self.patterns:
                    veto_fields = self._find_veto_fields(good_type, bad_type)
                    if veto_fields:
                        veto_patterns[f"{good_type}_blocked_by"] = veto_fields
        
        return veto_patterns
    
    def _find_veto_fields(self, good_type: str, bad_type: str) -> Dict:
        """Поиск полей блокирующих хорошие сигналы"""
        good_stats = self.patterns[good_type]['indicator_stats']
        bad_stats = self.patterns[bad_type]['indicator_stats']
        
        veto_fields = {}
        
        for indicator in set(good_stats.keys()) | set(bad_stats.keys()):
            good_activation = good_stats.get(indicator, {}).get('activation_rate', 0)
            bad_activation = bad_stats.get(indicator, {}).get('activation_rate', 0)
            
            # VETO поле: активно в плохих событиях, неактивно в хороших
            if bad_activation > 0.7 and good_activation < 0.3:
                veto_fields[indicator] = {
                    'good_activation': good_activation,
                    'bad_activation': bad_activation,
                    'veto_strength': bad_activation - good_activation
                }
        
        return veto_fields
    
    def generate_simple_tables(self) -> Dict:
        """Создание простых таблиц 'индикатор → тип события'"""
        print("📋 Создание простых таблиц...")
        
        if not self.patterns:
            return {}
        
        tables = {}
        
        # Топ индикаторы по каждому типу события
        for event_type, pattern_data in self.patterns.items():
            indicator_stats = pattern_data['indicator_stats']
            
            # Сортируем по важности (активация + абсолютное значение среднего)
            ranked_indicators = []
            for indicator, stats in indicator_stats.items():
                if 'mean' in stats and 'activation_rate' in stats:
                    importance = stats['activation_rate'] * (1 + abs(stats['mean']))
                    ranked_indicators.append({
                        'indicator': indicator,
                        'mean': stats['mean'],
                        'activation_rate': stats['activation_rate'],
                        'importance': importance,
                        'activations': stats['total_activations']
                    })
            
            # Сортируем по важности
            ranked_indicators.sort(key=lambda x: x['importance'], reverse=True)
            
            tables[event_type] = ranked_indicators[:20]  # Топ 20
        
        return tables
    
    def save_results(self, output_dir: str = "results/dump_pump_analysis"):
        """Сохранение результатов анализа"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Сохранение результатов в {output_path}")
        
        # Сохраняем события
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
            print(f"   ✅ Сохранена сводка событий: {len(events_df)} записей")
        
        # Сохраняем паттерны
        if self.patterns:
            with open(output_path / "patterns_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ Сохранен анализ паттернов")
        
        # Создаем простые таблицы
        simple_tables = self.generate_simple_tables()
        if simple_tables:
            with open(output_path / "simple_tables.json", 'w', encoding='utf-8') as f:
                json.dump(simple_tables, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ Сохранены простые таблицы")
        
        # Дискриминативные паттерны
        discriminative = self.find_discriminative_patterns()
        if discriminative:
            with open(output_path / "discriminative_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(discriminative, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ Сохранены дискриминативные паттерны")
        
        # VETO паттерны
        veto_patterns = self.find_veto_patterns()
        if veto_patterns:
            with open(output_path / "veto_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(veto_patterns, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ Сохранены VETO паттерны")
        
        # Создаем понятный отчет
        self._create_readable_report(output_path)
        
        return output_path
    
    def _create_readable_report(self, output_path: Path):
        """Создание понятного отчета"""
        report_lines = [
            "# АНАЛИЗ ПАТТЕРНОВ ДАМП/ПАМП",
            "=" * 50,
            "",
            f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Всего записей: {len(self.data) if self.data is not None else 0}",
            f"Найдено событий: {len(self.events)}",
            "",
            "## ТИПЫ СОБЫТИЙ",
            ""
        ]
        
        # Статистика по типам событий
        if self.events:
            event_counts = {}
            for event in self.events:
                event_type = event['type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            for event_type, count in sorted(event_counts.items()):
                report_lines.append(f"{event_type}: {count} событий")
        
        # Топ индикаторы по типам
        if self.patterns:
            simple_tables = self.generate_simple_tables()
            
            for event_type, indicators in simple_tables.items():
                report_lines.extend([
                    "",
                    f"## ТОП ИНДИКАТОРЫ ДЛЯ {event_type.upper()}",
                    ""
                ])
                
                for i, indicator_data in enumerate(indicators[:10], 1):
                    indicator = indicator_data['indicator']
                    mean = indicator_data['mean']
                    activation = indicator_data['activation_rate']
                    
                    report_lines.append(
                        f"{i:2d}. {indicator:15s} | "
                        f"среднее: {mean:8.2f} | "
                        f"активация: {activation:5.1%} | "
                        f"событий: {indicator_data['activations']}"
                    )
        
        # Дискриминативные паттерны
        discriminative = self.find_discriminative_patterns()
        if discriminative:
            report_lines.extend([
                "",
                "## РАЗЛИЧАЮЩИЕ ПАТТЕРНЫ",
                ""
            ])
            
            for comparison, patterns in discriminative.items():
                report_lines.append(f"### {comparison}")
                
                for indicator, data in list(patterns.items())[:5]:
                    report_lines.append(
                        f"  {indicator}: сила различия = {data['discriminative_power']:.2f}"
                    )
                report_lines.append("")
        
        # Сохраняем отчет
        with open(output_path / "readable_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   ✅ Создан понятный отчет")


def main():
    """Основная функция для анализа дамп/памп паттернов"""
    print("🚀 ЗАПУСК АНАЛИЗА ДАМП/ПАМП ПАТТЕРНОВ")
    print("=" * 60)
    
    # Инициализация анализатора
    analyzer = DumpPumpAnalyzer()
    
    # Путь к данным
    data_file = "data/dslog_btc_0508240229_ltf.txt"
    
    try:
        # 1. Загрузка и парсинг данных
        print("\n1️⃣ ЗАГРУЗКА ДАННЫХ")
        data = analyzer.load_and_parse_data(data_file)
        
        # 2. Поиск событий
        print("\n2️⃣ ПОИСК СОБЫТИЙ")
        events = analyzer.detect_events()
        
        if not events:
            print("❌ События не найдены, завершение работы")
            return
        
        # 3. Анализ паттернов
        print("\n3️⃣ АНАЛИЗ ПАТТЕРНОВ")
        patterns = analyzer.analyze_patterns()
        
        # 4. Дискриминативный анализ
        print("\n4️⃣ ДИСКРИМИНАТИВНЫЙ АНАЛИЗ")
        discriminative = analyzer.find_discriminative_patterns()
        
        # 5. VETO анализ
        print("\n5️⃣ VETO АНАЛИЗ")
        veto_patterns = analyzer.find_veto_patterns()
        
        # 6. Сохранение результатов
        print("\n6️⃣ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        output_path = analyzer.save_results()
        
        print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📁 Результаты сохранены в: {output_path}")
        print(f"📋 Проверьте файл: {output_path}/readable_report.txt")
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
