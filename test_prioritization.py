#!/usr/bin/env python3
"""
ТЕСТ ИСПРАВЛЕНИЯ ПРИОРИТИЗАЦИИ ИНДИКАТОРОВ
Проверяем работает ли радикальное исправление в parser_integration.py
"""

import sys
sys.path.append('.')

from pathlib import Path
import pandas as pd

def test_prioritization():
    """Тест исправления приоритизации"""
    print("🧪 ТЕСТ ИСПРАВЛЕНИЯ ПРИОРИТИЗАЦИИ ИНДИКАТОРОВ")
    print("=" * 60)
    
    try:
        # Импортируем исправленный модуль
        from parser_integration import ParserIntegration
        
        print("✅ Исправленный parser_integration.py загружен")
        
        # Проверяем есть ли тестовые данные
        test_file = Path("data/dslog_btc_0508240229_ltf.txt")
        if not test_file.exists():
            print(f"❌ Файл данных не найден: {test_file}")
            return False
        
        print(f"✅ Файл данных найден: {test_file}")
        
        # Создаем интеграцию
        integration = ParserIntegration()
        print("✅ ParserIntegration создан")
        
        # Проверяем веса приоритета
        print("\n📊 ПРОВЕРКА ВЕСОВ ПРИОРИТЕТА:")
        for group, weight in integration.group_priority_weights.items():
            print(f"   {group}: вес x{weight}")
        
        # Запускаем парсинг
        print("\n🔄 Запуск парсинга с исправленной приоритизацией...")
        results = integration.replace_old_parser(str(test_file))
        
        if not results:
            print("❌ Парсинг неуспешен")
            return False
        
        print("✅ Парсинг успешен")
        
        # Создаем признаки
        print("\n🔧 Создание признаков с приоритизацией...")
        features = integration.get_features_for_main_system()
        
        if features.empty:
            print("❌ Признаки не созданы")
            return False
        
        print(f"✅ Создано {len(features.columns)} признаков")
        
        # Анализируем результат приоритизации
        print("\n📊 АНАЛИЗ РЕЗУЛЬТАТА ПРИОРИТИЗАЦИИ:")
        
        priority_cols = [col for col in features.columns if col.startswith('PRIORITY_')]
        lowpri_cols = [col for col in features.columns if col.startswith('LOWPRI_')]
        derivative_cols = [col for col in features.columns if any(x in col for x in ['_LAG', '_DIFF', '_MA'])]
        
        print(f"   🔥 ПРИОРИТЕТНЫХ признаков: {len(priority_cols)}")
        print(f"   🔄 ПРОИЗВОДНЫХ признаков: {len(derivative_cols)}")
        print(f"   📋 НИЗКОПРИОРИТЕТНЫХ: {len(lowpri_cols)}")
        
        # Показываем примеры приоритетных признаков
        if priority_cols:
            print(f"\n🎯 ПРИМЕРЫ ПРИОРИТЕТНЫХ ПРИЗНАКОВ:")
            for i, col in enumerate(priority_cols[:5]):
                print(f"   {i+1}. {col}")
        
        # Показываем примеры низкоприоритетных
        if lowpri_cols:
            print(f"\n📋 ПРИМЕРЫ НИЗКОПРИОРИТЕТНЫХ (метаданные):")
            for i, col in enumerate(lowpri_cols[:3]):
                print(f"   {i+1}. {col}")
        
        # Проверяем соотношение
        total_indicator_features = len(priority_cols) + len(derivative_cols)
        total_metadata_features = len(lowpri_cols)
        
        print(f"\n📈 ИТОГОВОЕ СООТНОШЕНИЕ:")
        print(f"   Индикаторных признаков: {total_indicator_features}")
        print(f"   Метаданных: {total_metadata_features}")
        
        if total_metadata_features > 0:
            ratio = total_indicator_features / total_metadata_features
            print(f"   СООТНОШЕНИЕ: {ratio:.1f}:1")
            
            if ratio > 3:
                print("   ✅ ОТЛИЧНО: Индикаторы значительно доминируют!")
                success = True
            elif ratio > 1:
                print("   ⚡ ХОРОШО: Индикаторы преобладают")
                success = True
            else:
                print("   ❌ ПЛОХО: Метаданные все еще доминируют")
                success = False
        else:
            print("   ✅ ИДЕАЛЬНО: Только индикаторные признаки!")
            success = True
        
        # Отчет об интеграции
        print("\n" + "="*60)
        print(integration.integration_report())
        
        return success
        
    except Exception as e:
        print(f"❌ ОШИБКА ТЕСТА: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_prioritization()
    
    print("\n" + "="*60)
    if success:
        print("🎊 ТЕСТ ПРИОРИТИЗАЦИИ ПРОЙДЕН УСПЕШНО!")
        print("✅ Индикаторные поля получили максимальный приоритет")
        print("✅ Метаданные отправлены на минимальный приоритет")
        print("✅ Проблема доминирования метаданных РЕШЕНА!")
    else:
        print("❌ ТЕСТ ПРИОРИТИЗАЦИИ НЕ ПРОЙДЕН")
        print("⚠️ Требуется дополнительная настройка")
    print("="*60)

if __name__ == "__main__":
    main()
