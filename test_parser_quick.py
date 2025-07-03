#!/usr/bin/env python3
"""
БЫСТРЫЙ ТЕСТ ПАРСЕРА - ПРОВЕРЯЕМ РАБОТАЕТ ЛИ ОН
"""

import sys
import os
import re

# Добавляем путь к папке с модулями
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_log_parser import AdvancedLogParser
    print("✅ ИМПОРТ ПАРСЕРА УСПЕШЕН")
except ImportError as e:
    print(f"❌ ОШИБКА ИМПОРТА: {e}")
    sys.exit(1)

def test_parser():
    """Быстрый тест парсера"""
    
    # Тестовая строка с критическими полями  
    test_line = """[2024-08-05T09:24:00.000+03:00]: LTF|event_2025-06-28_22-55|1|2024-08-05 06:24|RED|-1.79%|11.5K|BIG_BODY|66%|-18.8%_24h|o:50254.8|h:50258.6|l:48888|c:49353.4|rng:1370.6|ef2--7.19|as2-3.33|vc2-2.4|nw2-!!|ze2--4.12|co2--213|ro2-11|so2-11"""
    
    print("\n🧪 ТЕСТ ПАРСЕРА")
    print("=" * 40)
    
    # Создаем парсер
    parser = AdvancedLogParser()
    
    # Парсим строку
    record = parser._parse_single_line(test_line, 0)
    
    if not record:
        print("❌ ПАРСЕР НЕ ИЗВЛЕК ДАННЫЕ")
        return False
    
    print(f"✅ Извлечено полей: {len(record)}")
    
    # Проверяем критические поля
    critical_tests = [
        ('ef2', -7.19),
        ('as2', 3.33), 
        ('vc2', 2.4),
        ('nw2', 2),  # Длина строки !!
        ('ze2', -4.12),
        ('co2', -213),
        ('ro2', 11),
        ('so2', 11)
    ]
    
    print("\n🎯 ПРОВЕРКА КРИТИЧЕСКИХ ПОЛЕЙ:")
    passed = 0
    
    for field, expected in critical_tests:
        if field in record:
            actual = record[field]
            if isinstance(expected, float):
                match = abs(actual - expected) < 0.01
            else:
                match = actual == expected
                
            if match:
                print(f"   ✅ {field}: {actual} (ожидалось {expected})")
                passed += 1
            else:
                print(f"   ❌ {field}: {actual} (ожидалось {expected})")
        else:
            print(f"   ❌ {field}: НЕ НАЙДЕНО")
    
    print(f"\nПройдено тестов: {passed}/{len(critical_tests)}")
    
    # Показываем все ef поля
    print(f"\n🔥 EF ПОЛЯ:")
    ef_fields = [k for k in record.keys() if k.startswith('ef') and not k.endswith('_type')]
    for field in ef_fields:
        print(f"   {field}: {record[field]}")
    
    # Показываем все nw поля
    print(f"\n🚨 NW ПОЛЯ:")
    nw_fields = [k for k in record.keys() if k.startswith('nw')]
    for field in nw_fields:
        print(f"   {field}: {record[field]}")
    
    return passed >= 6  # Минимум 6 из 8 полей должны работать

def test_file_parsing():
    """Тест парсинга файла"""
    
    log_file = "data/dslog_btc_0508240229_ltf.txt"
    
    if not os.path.exists(log_file):
        print(f"❌ Файл не найден: {log_file}")
        return False
    
    print(f"\n📁 ТЕСТ ПАРСИНГА ФАЙЛА")
    print("=" * 40)
    
    parser = AdvancedLogParser()
    df = parser.parse_log_file(log_file)
    
    if df.empty:
        print("❌ ФАЙЛ НЕ СПАРСИЛСЯ")
        return False
    
    print(f"✅ Извлечено записей: {len(df)}")
    print(f"✅ Извлечено полей: {len(df.columns)}")
    
    # Критические поля
    critical_fields = ['ef2', 'as2', 'vc2', 'nw2', 'ze2']
    found_critical = sum(1 for field in critical_fields if field in df.columns)
    
    print(f"✅ Критических полей найдено: {found_critical}/{len(critical_fields)}")
    
    # Показываем примеры значений
    print(f"\n📊 ПРИМЕРЫ ЗНАЧЕНИЙ:")
    for field in critical_fields:
        if field in df.columns:
            non_null = df[field].dropna()
            if len(non_null) > 0:
                print(f"   {field}: активаций={len(non_null)}, пример={non_null.iloc[0]}")
            else:
                print(f"   {field}: поле есть, но все NULL")
        else:
            print(f"   {field}: НЕ НАЙДЕНО")
    
    return found_critical >= 4

if __name__ == "__main__":
    print("🚀 БЫСТРЫЙ ТЕСТ ПАРСЕРА")
    print("=" * 50)
    
    # Тест 1: парсинг строки
    test1_passed = test_parser()
    
    # Тест 2: парсинг файла  
    test2_passed = test_file_parsing()
    
    print(f"\n🏆 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    print(f"   Тест строки: {'✅ ПРОЙДЕН' if test1_passed else '❌ ПРОВАЛЕН'}")
    print(f"   Тест файла: {'✅ ПРОЙДЕН' if test2_passed else '❌ ПРОВАЛЕН'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ПАРСЕР РАБОТАЕТ ПРАВИЛЬНО!")
        print(f"   Можно запускать скальп анализатор")
    else:
        print(f"\n💥 ПАРСЕР НЕ РАБОТАЕТ!")
        print(f"   Нужна отладка advanced_log_parser.py")
