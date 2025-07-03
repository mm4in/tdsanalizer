#!/usr/bin/env python3
"""
ПРОСТОЙ АНАЛИЗАТОР КОНТРТРЕНДОВОГО СКАЛЬПА
Ищет паттерны индикаторов для:
1. ЛОИ с откатами ВВЕРХ 3%+ (дамп → контртренд покупка)
2. ХАИ с откатами ВНИЗ 3%+ (памп → контртренд продажа)
3. Продолжения дампа/пампа (без откатов)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_log_parser import AdvancedLogParser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ScalpAnalyzer:
    """Простой анализатор для контртрендового скальпа"""
    
    def __init__(self):
        self.parser = AdvancedLogParser()
        self.df = None
        self.events = []
        
    def analyze_log(self, log_file: str):
        """Основной анализ лога"""
        print("🚀 АНАЛИЗ КОНТРТРЕНДОВОГО СКАЛЬПА")
        print("=" * 50)
        
        # 1. Парсим данные
        print("\n1️⃣ ПАРСИНГ ДАННЫХ...")
        self.df = self.parser.parse_log_file(log_file)
        
        if self.df.empty:
            print("❌ Данные не извлечены!")
            return
        
        # 2. Ищем события
        print("\n2️⃣ ПОИСК СОБЫТИЙ...")
        self.find_events()
        
        # 3. Анализируем паттерны
        print("\n3️⃣ АНАЛИЗ ПАТТЕРНОВ...")
        self.analyze_patterns()
        
        # 4. Создаем отчет
        print("\n4️⃣ СОЗДАНИЕ ОТЧЕТА...")
        self.create_simple_report()
        
    def find_events(self):
        """Поиск событий: лои/хаи с откатами vs продолжения"""
        
        # Вычисляем локальные минимумы и максимумы
        window = 10  # окно для поиска экстремумов
        
        self.df['is_local_low'] = (
            (self.df['low'] == self.df['low'].rolling(window=window, center=True).min()) &
            (self.df['low'].shift(1) > self.df['low']) &
            (self.df['low'].shift(-1) > self.df['low'])
        )
        
        self.df['is_local_high'] = (
            (self.df['high'] == self.df['high'].rolling(window=window, center=True).max()) &
            (self.df['high'].shift(1) < self.df['high']) &
            (self.df['high'].shift(-1) < self.df['high'])
        )
        
        # Ищем откаты после экстремумов
        events = []
        
        # Находим лои с откатами вверх
        local_lows = self.df[self.df['is_local_low']].copy()
        
        for idx, low_candle in local_lows.iterrows():
            # Ищем откат вверх в следующих 30 свечах
            future_data = self.df.loc[idx:idx+30]
            if len(future_data) < 10:
                continue
                
            max_price_after = future_data['high'].max()
            rebound_pct = ((max_price_after - low_candle['low']) / low_candle['low']) * 100
            
            if rebound_pct >= 3.0:  # Откат вверх 3%+
                events.append({
                    'type': 'ЛОЙ_КОНТРТРЕНД',
                    'timestamp': low_candle['timestamp'],
                    'price': low_candle['low'],
                    'rebound_pct': rebound_pct,
                    'line_number': idx
                })
            elif rebound_pct < 1.0:  # Продолжение дампа
                events.append({
                    'type': 'ПРОДОЛЖЕНИЕ_ДАМПА',
                    'timestamp': low_candle['timestamp'],
                    'price': low_candle['low'],
                    'rebound_pct': rebound_pct,
                    'line_number': idx
                })
        
        # Находим хаи с откатами вниз
        local_highs = self.df[self.df['is_local_high']].copy()
        
        for idx, high_candle in local_highs.iterrows():
            # Ищем откат вниз в следующих 30 свечах
            future_data = self.df.loc[idx:idx+30]
            if len(future_data) < 10:
                continue
                
            min_price_after = future_data['low'].min()
            pullback_pct = ((high_candle['high'] - min_price_after) / high_candle['high']) * 100
            
            if pullback_pct >= 3.0:  # Откат вниз 3%+
                events.append({
                    'type': 'ХАЙ_КОНТРТРЕНД',
                    'timestamp': high_candle['timestamp'],
                    'price': high_candle['high'],
                    'pullback_pct': pullback_pct,
                    'line_number': idx
                })
            elif pullback_pct < 1.0:  # Продолжение пампа
                events.append({
                    'type': 'ПРОДОЛЖЕНИЕ_ПАМПА',
                    'timestamp': high_candle['timestamp'],
                    'price': high_candle['high'],
                    'pullback_pct': pullback_pct,
                    'line_number': idx
                })
        
        self.events = events
        
        # Статистика событий
        event_counts = {}
        for event in events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print("📊 НАЙДЕННЫЕ СОБЫТИЯ:")
        for event_type, count in event_counts.items():
            print(f"   {event_type}: {count} событий")
        
        print(f"   ВСЕГО: {len(events)} событий")
    
    def analyze_patterns(self):
        """Анализ паттернов индикаторов перед событиями"""
        
        if not self.events:
            print("❌ События не найдены для анализа")
            return
        
        # Ключевые индикаторы для анализа
        key_indicators = ['nw2', 'ef2', 'as2', 'vc2', 'ze2', 'co2', 'ro2', 'so2']
        
        patterns = {}
        
        for event in self.events:
            event_type = event['type']
            line_num = event['line_number']
            
            if event_type not in patterns:
                patterns[event_type] = {indicator: [] for indicator in key_indicators}
            
            # Смотрим индикаторы за 5 свечей до события
            start_idx = max(0, line_num - 5)
            end_idx = line_num
            
            event_data = self.df.iloc[start_idx:end_idx]
            
            for indicator in key_indicators:
                if indicator in self.df.columns:
                    # Берем последнее непустое значение
                    values = event_data[indicator].dropna()
                    if len(values) > 0:
                        last_value = values.iloc[-1]
                        patterns[event_type][indicator].append(last_value)
        
        # Вычисляем статистики по паттернам
        self.pattern_stats = {}
        
        for event_type, indicators in patterns.items():
            self.pattern_stats[event_type] = {}
            
            for indicator, values in indicators.items():
                if values:
                    self.pattern_stats[event_type][indicator] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }
        
        print("✅ Анализ паттернов завершен")
    
    def create_simple_report(self):
        """Создание простого отчета с таблицами"""
        
        print("\n📋 ПРОСТЫЕ ТАБЛИЦЫ 'ИНДИКАТОР → ТИП СОБЫТИЯ':")
        print("=" * 60)
        
        if not hasattr(self, 'pattern_stats') or not self.pattern_stats:
            print("❌ Нет данных для отчета")
            return
        
        # Создаем простую таблицу
        event_types = list(self.pattern_stats.keys())
        indicators = ['nw2', 'ef2', 'as2', 'vc2', 'ze2', 'co2', 'ro2', 'so2']
        
        print(f"\n🎯 СРЕДНИЕ ЗНАЧЕНИЯ ИНДИКАТОРОВ:")
        print(f"{'ИНДИКАТОР':<10} | ", end="")
        for event_type in event_types:
            print(f"{event_type[:15]:<15} | ", end="")
        print()
        print("-" * (10 + 17 * len(event_types)))
        
        for indicator in indicators:
            print(f"{indicator:<10} | ", end="")
            
            for event_type in event_types:
                if (event_type in self.pattern_stats and 
                    indicator in self.pattern_stats[event_type]):
                    
                    mean_val = self.pattern_stats[event_type][indicator]['mean']
                    count = self.pattern_stats[event_type][indicator]['count']
                    print(f"{mean_val:>6.2f}({count:>2})<7 | ", end="")
                else:
                    print(f"{'N/A':>13} | ", end="")
            print()
        
        # Поиск потенциальных VETO полей
        print(f"\n🚫 ПОТЕНЦИАЛЬНЫЕ VETO ПОЛЯ:")
        
        # Поля, которые могут блокировать ложные сигналы
        veto_candidates = []
        
        for indicator in indicators:
            values_by_type = {}
            
            for event_type in event_types:
                if (event_type in self.pattern_stats and 
                    indicator in self.pattern_stats[event_type]):
                    values_by_type[event_type] = self.pattern_stats[event_type][indicator]['mean']
            
            # Если значения сильно различаются между типами событий
            if len(values_by_type) >= 2:
                values = list(values_by_type.values())
                value_range = max(values) - min(values)
                
                if value_range > 2.0:  # Порог различия
                    veto_candidates.append({
                        'indicator': indicator,
                        'range': value_range,
                        'values': values_by_type
                    })
        
        if veto_candidates:
            for candidate in sorted(veto_candidates, key=lambda x: x['range'], reverse=True):
                print(f"   {candidate['indicator']}: диапазон {candidate['range']:.2f}")
                for event_type, value in candidate['values'].items():
                    print(f"      {event_type}: {value:.2f}")
        else:
            print("   Потенциальные VETO поля не найдены")
        
        # Сохраняем детальный отчет в файл
        report_file = "scalp_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО КОНТРТРЕНДОВОМУ СКАЛЬПУ\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("СОБЫТИЯ:\n")
            for event in self.events[:20]:  # Первые 20 событий
                f.write(f"{event['type']}: {event['timestamp']} | {event['price']:.2f}\n")
            
            f.write(f"\nВСЕГО СОБЫТИЙ: {len(self.events)}\n")
            
            f.write("\nПАТТЕРНЫ ИНДИКАТОРОВ:\n")
            for event_type, indicators in self.pattern_stats.items():
                f.write(f"\n{event_type}:\n")
                for indicator, stats in indicators.items():
                    f.write(f"  {indicator}: mean={stats['mean']:.2f}, count={stats['count']}\n")
        
        print(f"\n💾 Детальный отчет сохранен: {report_file}")


def main():
    """Главная функция"""
    # Путь к логу
    log_file = "C:/Users/maksi/Documents/Claude fs/financial_analyzer/data/dslog_btc_0508240229_ltf.txt"
    
    if not os.path.exists(log_file):
        print(f"❌ Файл не найден: {log_file}")
        return
    
    # Создаем анализатор и запускаем
    analyzer = ScalpAnalyzer()
    analyzer.analyze_log(log_file)

if __name__ == "__main__":
    main()
