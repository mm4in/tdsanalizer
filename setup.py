#!/usr/bin/env python3
"""
Автоматическая настройка проекта "Финансовый анализатор логов"
Проверяет зависимости, создает структуру папок, генерирует тестовые данные
"""

import sys
import subprocess
import importlib
import json
import yaml
from pathlib import Path
import platform
import os

class ProjectSetup:
    """Класс для автоматической настройки проекта"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.project_dir = Path.cwd()
        self.errors = []
        self.warnings = []
        
    def check_python_version(self):
        """Проверка версии Python"""
        print("🐍 Проверка версии Python...")
        
        if self.python_version < (3, 8):
            self.errors.append(f"Требуется Python 3.8+, установлен {sys.version}")
            print(f"❌ Python {sys.version}")
            return False
        else:
            print(f"✅ Python {sys.version}")
            return True
    
    def check_pip(self):
        """Проверка наличия pip"""
        print("📦 Проверка pip...")
        
        try:
            import pip
            print("✅ pip установлен")
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                             check=True, capture_output=True)
                print("✅ pip доступен через модуль")
                return True
            except subprocess.CalledProcessError:
                self.errors.append("pip не установлен")
                print("❌ pip не найден")
                return False
    
    def install_dependencies(self):
        """Установка зависимостей"""
        print("📚 Установка зависимостей...")
        
        requirements_file = self.project_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print("⚠️ requirements.txt не найден, создаю...")
            self.create_requirements_file()
        
        try:
            # Обновление pip
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], check=True, capture_output=True)
            
            # Установка зависимостей
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], check=True, capture_output=True)
            
            print("✅ Зависимости установлены")
            return True
            
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Ошибка установки зависимостей: {e}")
            print("❌ Ошибка установки зависимостей")
            return False
    
    def create_requirements_file(self):
        """Создание файла requirements.txt"""
        requirements_content = """# Основные библиотеки для анализа данных
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Машинное обучение
scikit-learn>=1.1.0

# Работа с конфигурацией
PyYAML>=6.0

# Дополнительные библиотеки
python-dateutil>=2.8.0
tqdm>=4.64.0
scipy>=1.9.0
plotly>=5.10.0
"""
        
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
    
    def verify_imports(self):
        """Проверка импорта ключевых библиотек"""
        print("🔍 Проверка импорта библиотек...")
        
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'sklearn', 'yaml', 'scipy'
        ]
        
        failed_imports = []
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {package}")
            except ImportError:
                failed_imports.append(package)
                print(f"   ❌ {package}")
        
        if failed_imports:
            self.errors.append(f"Не удалось импортировать: {', '.join(failed_imports)}")
            return False
        
        print("✅ Все библиотеки импортированы успешно")
        return True
    
    def create_project_structure(self):
        """Создание структуры проекта"""
        print("📁 Создание структуры проекта...")
        
        directories = [
            "data",
            "results",
            "results/plots",
            "results/reports",
            "results/type_1",
            "results/type_2", 
            "results/combined",
            "logs",
            "backup"
        ]
        
        for directory in directories:
            dir_path = self.project_dir / directory
            dir_path.mkdir(exist_ok=True)
            print(f"   📁 {directory}")
        
        # Создание .gitignore если его нет
        gitignore_path = self.project_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
financial_analyzer_env/

# Project specific
data/*.txt
!data/sample_*.txt
results/*
!results/.gitkeep
logs/*.log
backup/*
*.pkl
*.joblib

# OS
.DS_Store
Thumbs.db
"""
            with open(gitignore_path, "w") as f:
                f.write(gitignore_content)
            print("   📄 .gitignore")
        
        print("✅ Структура проекта создана")
        return True
    
    def create_default_config(self):
        """Создание конфигурации по умолчанию"""
        print("⚙️ Создание конфигурации...")
        
        config_path = self.project_dir / "config.yaml"
        
        if config_path.exists():
            print("   ⚠️ config.yaml уже существует, пропускаю")
            return True
        
        default_config = {
            'analysis': {
                'min_accuracy': 0.60,
                'min_lift': 1.5,
                'validation_split': 0.3,
                'cv_folds': 5,
                'max_features': 100
            },
            'event_detection': {
                'volatility_threshold': 2.0,
                'volume_threshold': 1.5,
                'price_change_threshold': 0.5,
                'extreme_quantile': 0.8,
                'min_event_strength': 1.0
            },
            'feature_selection': {
                'max_features': 50,
                'correlation_threshold': 0.8,
                'min_variance': 0.01,
                'max_lags': 10
            },
            'scoring': {
                'min_roc_auc': 0.55,
                'min_activations': 10,
                'threshold_method': 'percentile',
                'rf_n_estimators': 100,
                'rf_random_state': 42,
                'rf_max_depth': 10
            },
            'timeframes': {
                'type_1': [2, 5, 15, 30],
                'type_2': ['1H', '4H', 'D', 'W']
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print("   📄 config.yaml")
        print("✅ Конфигурация создана")
        return True
    
    def create_sample_data(self):
        """Создание примерных данных"""
        print("🎲 Создание тестовых данных...")
        
        sample_file = self.project_dir / "data" / "sample_log.txt"
        
        if sample_file.exists():
            print("   ⚠️ Тестовые данные уже существуют")
            return True
        
        try:
            # Импорт data_utils для генерации данных
            sys.path.append(str(self.project_dir))
            from data_utils import DataProcessor
            
            processor = DataProcessor()
            processor.generate_sample_data(str(sample_file), num_records=150)
            
            print("   📄 sample_log.txt")
            print("✅ Тестовые данные созданы")
            return True
            
        except Exception as e:
            self.warnings.append(f"Не удалось создать тестовые данные: {e}")
            print(f"   ⚠️ Ошибка: {e}")
            return False
    
    def create_launcher_scripts(self):
        """Создание скриптов запуска"""
        print("🚀 Создание скриптов запуска...")
        
        # Windows batch файл
        if self.platform == "Windows" or True:  # Создаем для всех платформ
            batch_content = """@echo off
echo Запуск финансового анализатора...
python main.py data\\sample_log.txt
pause
"""
            with open("quick_start.bat", "w", encoding="utf-8") as f:
                f.write(batch_content)
            print("   📄 quick_start.bat")
        
        # Unix shell script
        if self.platform in ["Linux", "Darwin"] or True:  # Создаем для всех платформ
            shell_content = """#!/bin/bash
echo "Запуск финансового анализатора..."
python3 main.py data/sample_log.txt
read -p "Нажмите Enter для выхода..."
"""
            shell_file = Path("quick_start.sh")
            with open(shell_file, "w", encoding="utf-8") as f:
                f.write(shell_content)
            
            # Права на выполнение для Unix
            if self.platform in ["Linux", "Darwin"]:
                shell_file.chmod(0o755)
            print("   📄 quick_start.sh")
        
        print("✅ Скрипты запуска созданы")
        return True
    
    def create_readme_summary(self):
        """Создание краткой инструкции"""
        print("📖 Создание краткой инструкции...")
        
        quick_start_content = """# 🚀 БЫСТРЫЙ СТАРТ

## Что делать после установки:

### 1. Базовый тест (рекомендуется)
```bash
python main.py data/sample_log.txt
```

### 2. TYPE-анализ (расширенный)
```bash
python type_analyzer.py data/sample_log.txt
```

### 3. Утилиты для данных
```bash
python data_utils.py sample data/my_test.txt -n 300
python data_utils.py validate data/my_log.txt
python data_utils.py clean data/my_log.txt
```

### 4. API для интеграции
```python
from scoring_api import ScoringAPI
api = ScoringAPI()
result = api.score_log_line("ваша_строка_лога")
```

## 📁 Структура результатов:
- `results/weight_matrix.csv` - матрица весов (главный результат!)
- `results/scoring_config.json` - конфигурация скоринга
- `results/plots/` - графики и визуализации
- `results/reports/` - детальные отчеты

## 🔧 Настройка:
Измените параметры в `config.yaml` под ваши нужды.

## ❓ Проблемы:
1. Проверьте `requirements.txt` - все ли библиотеки установлены
2. Используйте `python3` вместо `python` на Linux/Mac
3. Активируйте виртуальное окружение если используете

## 📞 Поддержка:
Создайте issue с описанием проблемы и приложите первые 10 строк вашего лога.
"""
        
        with open("QUICK_START.md", "w", encoding="utf-8") as f:
            f.write(quick_start_content)
        
        print("   📄 QUICK_START.md")
        print("✅ Краткая инструкция создана")
        return True
    
    def validate_installation(self):
        """Финальная валидация установки"""
        print("🔍 Финальная валидация...")
        
        # Проверка основных файлов
        required_files = [
            "main.py", "type_analyzer.py", "data_utils.py", 
            "scoring_api.py", "config.yaml", "requirements.txt"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.errors.append(f"Отсутствуют файлы: {', '.join(missing_files)}")
            return False
        
        # Тест импорта основного модуля
        try:
            sys.path.insert(0, str(self.project_dir))
            import main
            print("   ✅ main.py импортируется")
        except ImportError as e:
            self.errors.append(f"Ошибка импорта main.py: {e}")
            return False
        
        # Проверка тестовых данных
        sample_file = self.project_dir / "data" / "sample_log.txt"
        if sample_file.exists():
            print("   ✅ Тестовые данные доступны")
        else:
            self.warnings.append("Тестовые данные недоступны")
        
        print("✅ Валидация завершена")
        return True
    
    def run_setup(self):
        """Запуск полной настройки"""
        print("🛠️ АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПРОЕКТА")
        print("=" * 50)
        
        steps = [
            ("Проверка Python", self.check_python_version),
            ("Проверка pip", self.check_pip),
            ("Установка зависимостей", self.install_dependencies),
            ("Проверка импорта", self.verify_imports),
            ("Создание структуры", self.create_project_structure),
            ("Создание конфигурации", self.create_default_config),
            ("Создание тестовых данных", self.create_sample_data),
            ("Создание скриптов", self.create_launcher_scripts),
            ("Создание документации", self.create_readme_summary),
            ("Финальная валидация", self.validate_installation)
        ]
        
        completed_steps = 0
        
        for step_name, step_function in steps:
            print(f"\n📋 {step_name}...")
            try:
                if step_function():
                    completed_steps += 1
                else:
                    print(f"❌ {step_name} не выполнен")
            except Exception as e:
                self.errors.append(f"Ошибка в {step_name}: {e}")
                print(f"❌ {step_name}: {e}")
        
        # Итоговый отчет
        print("\n" + "=" * 50)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 50)
        
        print(f"✅ Выполнено шагов: {completed_steps}/{len(steps)}")
        
        if self.warnings:
            print(f"⚠️ Предупреждения ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.errors:
            print(f"❌ Ошибки ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
            print("\n💡 Рекомендации:")
            print("   1. Проверьте интернет-соединение")
            print("   2. Обновите pip: python -m pip install --upgrade pip")
            print("   3. Используйте виртуальное окружение")
            print("   4. Проверьте права доступа к папке")
        else:
            print("\n🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
            print("\n🚀 Следующие шаги:")
            print("   1. Запустите: python main.py data/sample_log.txt")
            print("   2. Проверьте результаты в папке results/")
            print("   3. Изучите QUICK_START.md для дополнительных возможностей")
            print("   4. Настройте config.yaml под ваши нужды")
            
            # Предложение запустить тест
            if completed_steps == len(steps):
                try:
                    response = input("\n❓ Запустить тестовый анализ сейчас? (y/n): ")
                    if response.lower() in ['y', 'yes', 'да']:
                        self.run_test_analysis()
                except KeyboardInterrupt:
                    print("\n⏹️ Отменено пользователем")
        
        return completed_steps == len(steps) and not self.errors
    
    def run_test_analysis(self):
        """Запуск тестового анализа"""
        print("\n🧪 Запуск тестового анализа...")
        
        try:
            sample_file = self.project_dir / "data" / "sample_log.txt"
            
            if not sample_file.exists():
                print("❌ Тестовые данные не найдены")
                return
            
            # Импорт и запуск анализатора
            sys.path.insert(0, str(self.project_dir))
            from main import FinancialLogAnalyzer
            
            print("📊 Создание анализатора...")
            analyzer = FinancialLogAnalyzer()
            
            print("🔍 Запуск анализа (это может занять 1-2 минуты)...")
            results = analyzer.run_full_analysis(str(sample_file))
            
            if results['status'] == 'success':
                print("✅ Тестовый анализ завершен успешно!")
                
                if 'validation_results' in results and results['validation_results']:
                    vr = results['validation_results']
                    print(f"   📈 ROC-AUC: {vr.get('roc_auc', 0):.3f}")
                    print(f"   🎯 Точность: {vr.get('accuracy', 0):.3f}")
                    print(f"   🚀 Lift: {vr.get('lift', 0):.3f}")
                
                print("   📁 Результаты сохранены в папке results/")
                print("   📊 Проверьте weight_matrix.csv - это ваша модель!")
            else:
                print(f"❌ Ошибка тестового анализа: {results.get('message', 'Неизвестная ошибка')}")
                
        except Exception as e:
            print(f"❌ Ошибка запуска тестового анализа: {e}")


def main():
    """Главная функция"""
    setup = ProjectSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Установка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка установки: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()