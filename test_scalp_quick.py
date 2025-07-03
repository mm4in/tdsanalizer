#!/usr/bin/env python3
"""
БЫСТРЫЙ ТЕСТ СКАЛЬП АНАЛИЗАТОРА - ПРОВЕРЯЕМ ВСЮ СИСТЕМУ
"""

import sys
import os
import pandas as pd
import numpy as np

# Добавляем путь к папке с модулями
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scalp_analyzer import ScalpAnalyzer
    print("✅ ИМПОРТ СКАЛЬП АНАЛИЗАТОРА УСПЕШЕН")
except ImportError as e:
    print(f"❌ ОШИБКА ИМПОРТА СКАЛЬП АНАЛИЗАТОРА: {e}")
    
    try:
        from advanced_log_parser import AdvancedLogParser
        print("✅ ИМПОРТ ПАРСЕРА УСПЕШЕН")
        print("⚠️ Будем тестировать только парсер")
        USE_FULL_ANALYZER = False
    except ImportError as e2:
        print(f"❌ ОШИБКА ИМПОРТА ПАРСЕРА: {e2}")
        sys.exit(1)
else:
    USE_FULL_ANALYZER = True

def test_full_scalp_analyzer():
    """Тест полного скальп анализатора"""
    
    log_file = "data/dslog_btc_0508240229_ltf.txt"
    
    if not os.path.exists(log_file):
        print(f"❌ Файл не найден: {log_file}")
        return False
    
    print(f"\n🚀 ТЕСТ ПОЛНОГО СКАЛЬП АНАЛИЗАТОРА")
    print("=" * 50)
    
    try:
        # Создаем анализатор
        analyzer = ScalpAnalyzer()
        
        # Запускаем анализ
        analyzer.analyze_log(log_file)
        
        # Проверяем результаты
        if analyzer.df is None or analyzer.df.empty:
            print("❌ ДАННЫЕ НЕ ЗАГРУЖЕНЫ")
            return False
        
        print(f"✅ Данные загружены: {len(analyzer.df)} записей, {len(analyzer.df.columns)} полей")
        
        # Проверяем события
        if not analyzer.events:
            print("❌ СОБЫТИЯ НЕ НАЙДЕНЫ") 
            return False
        
        print(f"✅ События найдены: {len(analyzer.events)}")
        
        # Статистика по типам событий
        event_types = {}
        for event in analyzer.events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print(f"\n📊 СТАТИСТИКА СОБЫТИЙ:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")
        
        # Проверяем паттерны
        if hasattr(analyzer, 'pattern_stats') and analyzer.pattern_stats:
            print(f"✅ Паттерны проанализированы")
        else:
            print(f"❌ ПАТТЕРНЫ НЕ ПРОАНАЛИЗИРОВАНЫ")
        
        # Проверяем создался ли отчет
        report_file = "scalp_analysis_report.txt"
        if os.path.exists(report_file):
            print(f"✅ Отчет создан: {report_file}")
        else:
            print(f"❌ ОТЧЕТ НЕ СОЗДАН")
        
        return True
        
    except Exception as e:
        print(f"❌ ОШИБКА ПРИ АНАЛИЗЕ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parser_only():
    """Тест только парсера (если полный анализатор не работает)"""
    
    log_file = "data/dslog_btc_0508240229_ltf.txt"
    
    if not os.path.exists(log_file):
        print(f"❌ Файл не найден: {log_file}")
        return False
    
    print(f"\n📊 ТЕСТ ТОЛЬКО ПАРСЕРА")
    print("=" * 30)
    
    from advanced_log_parser import AdvancedLogParser
    
    parser = AdvancedLogParser()
    df = parser.parse_log_file(log_file)
    
    if df.empty:
        print("❌ ПАРСЕР НЕ РАБОТАЕТ")
        return False
    
    print(f"✅ Парсер работает: {len(df)} записей, {len(df.columns)} полей")
    
    # Проверяем критические поля
    critical_fields = ['ef2', 'as2', 'vc2', 'nw2', 'ze2']
    found_critical = []
    
    for field in critical_fields:
        if field in df.columns:
            non_null_count = df[field].notna().sum()
            if non_null_count > 0:
                found_critical.append(field)
                print(f"   ✅ {field}: {non_null_count} активаций")
            else:
                print(f"   ⚠️ {field}: поле есть, но все NULL")
        else:
            print(f"   ❌ {field}: НЕ НАЙДЕНО")
    
    print(f"\nКритических полей найдено: {len(found_critical)}/{len(critical_fields)}")
    
    return len(found_critical) >= 3

def simple_event_detection(df):
    """Простой поиск событий если основной анализатор не работает"""
    
    if 'low' not in df.columns or 'high' not in df.columns:
        print("❌ НЕТ ДАННЫХ О ЦЕНАХ")
        return []
    
    print(f"\n🔍 ПРОСТОЙ ПОИСК СОБЫТИЙ")
    print("=" * 30)
    
    events = []
    window = 5  # Упрощенное окно
    
    # Ищем локальные минимумы
    for i in range(window, len(df) - window):
        current_low = df.iloc[i]['low']
        is_local_min = True
        
        # Проверяем является ли это локальным минимумом
        for j in range(i - window, i + window + 1):
            if j != i and df.iloc[j]['low'] <= current_low:
                is_local_min = False
                break
        
        if is_local_min:
            # Ищем откат в следующих 10 свечах
            future_slice = df.iloc[i:i+10]
            if len(future_slice) > 5:
                max_high = future_slice['high'].max()
                rebound_pct = ((max_high - current_low) / current_low) * 100
                
                if rebound_pct >= 3.0:
                    events.append({
                        'type': 'ЛОЙ_КОНТРТРЕНД',
                        'rebound_pct': rebound_pct,
                        'index': i
                    })
                elif rebound_pct < 1.0:
                    events.append({
                        'type': 'ПРОДОЛЖЕНИЕ_ДАМПА', 
                        'rebound_pct': rebound_pct,
                        'index': i
                    })
    
    print(f"Найдено событий: {len(events)}")
    
    # Статистика
    event_counts = {}
    for event in events:
        event_type = event['type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    for event_type, count in event_counts.items():
        print(f"   {event_type}: {count}")
    
    return events

if __name__ == "__main__":
    print("🚀 БЫСТРЫЙ ТЕСТ СКАЛЬП СИСТЕМЫ")
    print("=" * 60)
    
    if USE_FULL_ANALYZER:
        # Тест полного анализатора
        full_test_passed = test_full_scalp_analyzer()
        
        if full_test_passed:
            print(f"\n🎉 СКАЛЬП АНАЛИЗАТОР РАБОТАЕТ ПОЛНОСТЬЮ!")
        else:
            print(f"\n⚠️ ПРОБЛЕМЫ С ПОЛНЫМ АНАЛИЗАТОРОМ, ТЕСТИРУЕМ ПАРСЕР...")
            parser_test_passed = test_parser_only()
            
            if parser_test_passed:
                print(f"\n✅ ПАРСЕР РАБОТАЕТ, ПРОБЛЕМА В ЛОГИКЕ АНАЛИЗАТОРА")
            else:
                print(f"\n❌ ПАРСЕР НЕ РАБОТАЕТ - КРИТИЧЕСКАЯ ПРОБЛЕМА")
    else:
        # Тест только парсера
        parser_test_passed = test_parser_only()
        
        if parser_test_passed:
            print(f"\n✅ ПАРСЕР РАБОТАЕТ")
            
            # Пробуем простой поиск событий
            from advanced_log_parser import AdvancedLogParser
            parser = AdvancedLogParser() 
            df = parser.parse_log_file("data/dslog_btc_0508240229_ltf.txt")
            events = simple_event_detection(df)
            
            if events:
                print(f"✅ ПРОСТОЙ ПОИСК СОБЫТИЙ РАБОТАЕТ")
            else:
                print(f"❌ СОБЫТИЯ НЕ НАЙДЕНЫ")
        else:
            print(f"\n❌ ПАРСЕР НЕ РАБОТАЕТ")
    
    print(f"\n📋 ЗАКЛЮЧЕНИЕ:")
    if USE_FULL_ANALYZER and 'full_test_passed' in locals() and full_test_passed:
        print(f"   🎯 Система готова для продвинутого анализа!")
        print(f"   📊 Проверьте файл scalp_analysis_report.txt")
    elif 'parser_test_passed' in locals() and parser_test_passed:
        print(f"   🔧 Парсер работает, но нужна доработка логики")
        print(f"   📊 Критические поля извлекаются правильно")
    else:
        print(f"   🚨 СИСТЕМА НЕ РАБОТАЕТ - ТРЕБУЕТ ИСПРАВЛЕНИЯ")
