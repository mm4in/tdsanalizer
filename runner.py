#!/usr/bin/env python3
"""
Скрипт запуска ЧЕСТНОГО финансового анализатора
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Использование: python runner.py <путь_к_файлу_лога>")
        print("Пример: python runner.py data/dslog_btc_0508240229_ltf.txt")
        return
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"❌ Файл не найден: {log_file}")
        return
    
    # ИСПРАВЛЕНО: Импорт нового честного анализатора
    from main import HonestDataDrivenAnalyzer
    
    analyzer = HonestDataDrivenAnalyzer()
    results = analyzer.run_full_analysis(log_file)
    
    if results['status'] == 'success':
        print("\n" + "="*70)
        print("🎊 ЧЕСТНЫЙ DATA-DRIVEN АНАЛИЗ ЗАВЕРШЕН!")
        print("="*70)
        print(f"📁 Все результаты: {results['results_folder']}")
        print(f"📋 Главный отчет: {results['results_folder']}/СТАТИСТИЧЕСКИЙ_АНАЛИЗ.txt")
        if results.get('validation_results'):
            val = results['validation_results']
            print(f"📊 Качество модели: ROC-AUC {val['roc_auc']:.3f}")
        print("🎯 ПРИНЦИП: ДАННЫЕ САМИ ПОКАЗАЛИ ЧТО ВАЖНО!")
        print("="*70)
    else:
        print(f"❌ Ошибка: {results['message']}")

if __name__ == "__main__":
    main()