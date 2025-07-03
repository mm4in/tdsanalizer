#!/usr/bin/env python3
"""
Запуск анализа дамп/памп паттернов
"""

import sys
import os

# Добавляем текущую папку в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dump_pump_analyzer import DumpPumpAnalyzer

def run_analysis():
    """Запуск анализа"""
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
        
        # Покажем краткую информацию о данных
        print(f"📊 Загружено {len(data)} записей")
        print(f"📊 Столбцов: {len(data.columns)}")
        
        # Покажем примеры полей
        indicator_fields = [col for col in data.columns 
                           if not col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'range', 
                                        'candle_color', 'candle_type', 'line_number', 'raw_line']]
        print(f"📊 Индикаторных полей: {len(indicator_fields)}")
        print(f"📊 Примеры полей: {', '.join(indicator_fields[:10])}")
        
        # 2. Поиск событий
        print("\n2️⃣ ПОИСК СОБЫТИЙ")
        events = analyzer.detect_events()
        
        if not events:
            print("❌ События не найдены, завершение работы")
            return None
        
        # 3. Анализ паттернов
        print("\n3️⃣ АНАЛИЗ ПАТТЕРНОВ")
        patterns = analyzer.analyze_patterns()
        
        # 4. Сохранение результатов
        print("\n4️⃣ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        output_path = analyzer.save_results()
        
        print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📁 Результаты сохранены в: {output_path}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = run_analysis()
