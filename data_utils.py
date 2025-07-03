#!/usr/bin/env python3
"""
Утилиты для работы с финансовыми данными
Предобработка, валидация и конвертация логов
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

class DataProcessor:
    """Класс для предобработки и валидации данных"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.log', '.csv']
        self.validation_stats = {}
    
    def validate_log_format(self, file_path):
        """Валидация формата лог файла"""
        print(f"🔍 Валидация файла: {file_path}")
        
        validation_results = {
            'file_exists': False,
            'readable': False,
            'correct_format': False,
            'line_count': 0,
            'valid_lines': 0,
            'timestamp_format': False,
            'required_fields': False,
            'errors': []
        }
        
        file_path = Path(file_path)
        
        # Проверка существования файла
        if not file_path.exists():
            validation_results['errors'].append("Файл не найден")
            return validation_results
        
        validation_results['file_exists'] = True
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            validation_results['readable'] = True
            validation_results['line_count'] = len(lines)
            
        except Exception as e:
            validation_results['errors'].append(f"Ошибка чтения файла: {e}")
            return validation_results
        
        # Анализ строк
        valid_line_count = 0
        timestamp_valid = False
        required_fields_found = False
        
        for i, line in enumerate(lines[:100]):  # Проверяем первые 100 строк
            line = line.strip()
            if not line:
                continue
            
            # Проверка формата timestamp
            timestamp_match = re.match(r'\[([^\]]+)\]:', line)
            if timestamp_match:
                try:
                    pd.to_datetime(timestamp_match.group(1))
                    timestamp_valid = True
                except:
                    pass
            
            # Проверка основного формата
            if '|' in line and 'LTF|' in line:
                parts = line.split('|')
                if len(parts) >= 6:
                    valid_line_count += 1
                    
                    # Проверка обязательных полей
                    if 'o:' in line and 'h:' in line and 'l:' in line and 'c:' in line:
                        required_fields_found = True
        
        validation_results['valid_lines'] = valid_line_count
        validation_results['timestamp_format'] = timestamp_valid
        validation_results['required_fields'] = required_fields_found
        validation_results['correct_format'] = (
            valid_line_count > 0 and timestamp_valid and required_fields_found
        )
        
        if validation_results['correct_format']:
            print("✅ Файл прошел валидацию")
        else:
            print("❌ Файл не прошел валидацию")
            for error in validation_results['errors']:
                print(f"   - {error}")
        
        return validation_results
    
    def clean_log_data(self, input_file, output_file=None):
        """Очистка и стандартизация лог данных"""
        print(f"🧹 Очистка данных: {input_file}")
        
        if output_file is None:
            output_file = str(Path(input_file).with_suffix('.cleaned.txt'))
        
        cleaned_lines = []
        stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'cleaned_lines': 0,
            'removed_duplicates': 0,
            'fixed_timestamps': 0
        }
        
        seen_lines = set()
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Удаление дубликатов
                if line in seen_lines:
                    stats['removed_duplicates'] += 1
                    continue
                seen_lines.add(line)
                
                # Базовая валидация формата
                if not (line.startswith('[') and 'LTF|' in line):
                    continue
                
                stats['valid_lines'] += 1
                
                # Очистка и стандартизация
                cleaned_line = self._clean_line(line)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
                    stats['cleaned_lines'] += 1
        
        # Сохранение очищенных данных
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"✅ Очистка завершена. Сохранено в: {output_file}")
        print(f"   Обработано строк: {stats['total_lines']}")
        print(f"   Валидных строк: {stats['valid_lines']}")
        print(f"   Очищенных строк: {stats['cleaned_lines']}")
        print(f"   Удалено дубликатов: {stats['removed_duplicates']}")
        
        return output_file, stats
    
    def _clean_line(self, line):
        """Очистка отдельной строки"""
        try:
            # Удаление лишних пробелов
            line = re.sub(r'\s+', ' ', line)
            
            # Стандартизация разделителей
            line = line.replace(' | ', '|').replace('| ', '|').replace(' |', '|')
            
            # Исправление common проблем с форматированием
            line = re.sub(r'([a-zA-Z]+)(\d+)--?([+-]?\d+\.?\d*)', r'\1\2-\3', line)
            
            return line
            
        except Exception:
            return None
    
    def convert_google_sheets_export(self, input_file, output_file=None):
        """Конвертация экспорта из Google Sheets"""
        print(f"📊 Конвертация из Google Sheets: {input_file}")
        
        if output_file is None:
            output_file = str(Path(input_file).with_suffix('.converted.txt'))
        
        # Попытка чтения как CSV
        try:
            df = pd.read_csv(input_file)
            
            # Предполагаем, что данные в определенных колонках
            converted_lines = []
            
            for _, row in df.iterrows():
                # Попытка реконструкции формата лога
                # Это зависит от структуры экспорта
                line_parts = [str(val) for val in row.values if pd.notna(val)]
                if len(line_parts) > 5:
                    # Попытка создания валидной строки
                    timestamp = pd.Timestamp.now().strftime('[%Y-%m-%dT%H:%M:%S.000+03:00]')
                    converted_line = f"{timestamp}: {' | '.join(line_parts)}"
                    converted_lines.append(converted_line)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(converted_lines))
            
            print(f"✅ Конвертация завершена: {output_file}")
            
        except Exception as e:
            print(f"❌ Ошибка конвертации: {e}")
            return None
        
        return output_file
    
    def split_by_events(self, input_file, output_dir=None):
        """Разделение лога по событиям"""
        print(f"✂️ Разделение по событиям: {input_file}")
        
        if output_dir is None:
            output_dir = Path(input_file).parent / 'split_events'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        current_event = None
        current_lines = []
        event_files = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Извлечение названия события
                match = re.search(r'LTF\|([^|]+)\|', line)
                if match:
                    event_name = match.group(1)
                    
                    # Если началось новое событие
                    if current_event != event_name:
                        # Сохранение предыдущего события
                        if current_event and current_lines:
                            event_file = output_dir / f"{current_event}.txt"
                            with open(event_file, 'w', encoding='utf-8') as ef:
                                ef.write('\n'.join(current_lines))
                            event_files.append(event_file)
                        
                        # Начало нового события
                        current_event = event_name
                        current_lines = []
                
                current_lines.append(line)
        
        # Сохранение последнего события
        if current_event and current_lines:
            event_file = output_dir / f"{current_event}.txt"
            with open(event_file, 'w', encoding='utf-8') as ef:
                ef.write('\n'.join(current_lines))
            event_files.append(event_file)
        
        print(f"✅ Создано {len(event_files)} файлов событий в {output_dir}")
        return event_files
    
    def merge_log_files(self, input_files, output_file):
        """Объединение нескольких лог файлов"""
        print(f"🔗 Объединение {len(input_files)} файлов")
        
        all_lines = []
        
        for file_path in input_files:
            print(f"   Обработка: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_lines.extend([line.strip() for line in lines if line.strip()])
        
        # Сортировка по timestamp если возможно
        try:
            def extract_timestamp(line):
                match = re.match(r'\[([^\]]+)\]:', line)
                if match:
                    return pd.to_datetime(match.group(1))
                return pd.Timestamp.min
            
            all_lines.sort(key=extract_timestamp)
            print("✅ Строки отсортированы по времени")
            
        except Exception as e:
            print(f"⚠️ Не удалось отсортировать по времени: {e}")
        
        # Удаление дубликатов
        unique_lines = list(dict.fromkeys(all_lines))
        removed_duplicates = len(all_lines) - len(unique_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_lines))
        
        print(f"✅ Объединение завершено: {output_file}")
        print(f"   Всего строк: {len(all_lines)}")
        print(f"   Уникальных строк: {len(unique_lines)}")
        print(f"   Удалено дубликатов: {removed_duplicates}")
        
        return output_file
    
    def generate_sample_data(self, output_file, num_records=200):
        """Генерация примерных данных для тестирования"""
        print(f"🎲 Генерация {num_records} тестовых записей")
        
        sample_lines = []
        base_time = pd.Timestamp('2024-08-05 09:00:00+03:00')
        
        # Группы полей для случайной генерации
        field_groups = {
            'group_1': ['rd', 'md', 'cd', 'cmd', 'macd'],
            'group_2': ['ro', 'mo', 'co', 'cz', 'do'],
            'group_3': ['rz', 'mz', 'ciz', 'sz', 'cvz'],
            'group_4': ['ef', 'wv', 'vc', 'ze', 'as'],
            'group_5': ['bs', 'wa', 'pd']
        }
        
        timeframes = [2, 5, 15, 30]
        
        for i in range(num_records):
            timestamp = base_time + pd.Timedelta(minutes=i)
            
            # Генерация OHLC данных
            base_price = 50000 + np.random.normal(0, 1000) * (1 + i/1000)  # Добавляем тренд
            volatility = np.random.uniform(0.5, 3.0)
            
            open_price = base_price + np.random.normal(0, 50)
            high_price = open_price + abs(np.random.exponential(50 * volatility))
            low_price = open_price - abs(np.random.exponential(50 * volatility))
            close_price = open_price + np.random.normal(0, 100 * volatility)
            
            # Свойства свечи
            color = "GREEN" if close_price > open_price else "RED"
            change = ((close_price - open_price) / open_price) * 100
            volume = f"{np.random.uniform(0.5, 15):.1f}K"
            
            candle_types = ["NORMAL", "BIG_BODY", "DOJI", "PIN_TOP", "PIN_BOTTOM"]
            candle_type = np.random.choice(candle_types, p=[0.6, 0.2, 0.1, 0.05, 0.05])
            
            completion = np.random.randint(10, 100)
            movement_24h = np.random.uniform(-25, 25)
            
            # Генерация полей
            fields = []
            
            # Событийная логика - больше активации перед "событиями"
            is_event_period = (i % 50) < 5  # Каждые 50 записей - 5 записей события
            activation_probability = 0.4 if is_event_period else 0.1
            
            for group_name, group_fields in field_groups.items():
                for field_base in group_fields:
                    if np.random.random() < activation_probability:
                        tf = np.random.choice(timeframes)
                        
                        if group_name == 'group_3':  # Z-scores
                            value = np.random.uniform(-4.0, 4.0)
                            fields.append(f"{field_base}{tf}--{value:.2f}")
                        elif group_name == 'group_1':  # Percentages
                            value = np.random.uniform(-50, 50)
                            fields.append(f"{field_base}{tf}-{value:.1f}%")
                        else:  # Regular values
                            value = np.random.randint(10, 100)
                            fields.append(f"{field_base}{tf}-{value}")
            
            # Добавление прогресса формирования
            progress_fields = []
            for tf in [2, 5, 15, 30]:
                if np.random.random() > 0.5:
                    progress = np.random.randint(0, 100)
                    progress_fields.append(f"p{tf}-{progress}")
            
            # Формирование строки лога
            log_line = (
                f"[{timestamp.strftime('%Y-%m-%dT%H:%M:%S.000+03:00')}]: "
                f"LTF|event_sample_{i // 50 + 1}|1|{timestamp.strftime('%Y-%m-%d %H:%M')}|"
                f"{color}|{change:.2f}%|{volume}|{candle_type}|{completion}%|{movement_24h:.2f}%_24h|"
                f"o:{open_price:.1f}|h:{high_price:.1f}|l:{low_price:.1f}|c:{close_price:.1f}|"
                f"rng:{high_price-low_price:.1f}"
            )
            
            # Добавление полей
            all_fields = fields + progress_fields
            if all_fields:
                log_line += "|" + ",".join(all_fields)
            
            sample_lines.append(log_line)
        
        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_lines))
        
        print(f"✅ Тестовые данные созданы: {output_file}")
        print(f"   Записей: {num_records}")
        print(f"   События: {num_records // 50}")
        print(f"   Временной диапазон: {sample_lines[0].split(']')[0][1:]} - {sample_lines[-1].split(']')[0][1:]}")
        
        return output_file
    
    def analyze_data_quality(self, file_path):
        """Анализ качества данных"""
        print(f"📊 Анализ качества данных: {file_path}")
        
        quality_report = {
            'file_info': {},
            'structure_analysis': {},
            'field_analysis': {},
            'temporal_analysis': {},
            'recommendations': []
        }
        
        # Базовая информация о файле
        file_path = Path(file_path)
        quality_report['file_info'] = {
            'file_size': file_path.stat().st_size,
            'created': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        # Анализ структуры
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        valid_lines = 0
        timestamps = []
        field_counts = {}
        all_fields = set()
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('['):
                continue
            
            valid_lines += 1
            
            # Извлечение timestamp
            timestamp_match = re.match(r'\[([^\]]+)\]:', line)
            if timestamp_match:
                try:
                    ts = pd.to_datetime(timestamp_match.group(1))
                    timestamps.append(ts)
                except:
                    pass
            
            # Подсчет полей
            field_matches = re.findall(r'([a-zA-Z]+\d*)-?([^,|]+)', line)
            line_fields = len(field_matches)
            field_counts[line_fields] = field_counts.get(line_fields, 0) + 1
            
            for field_name, _ in field_matches:
                all_fields.add(field_name)
        
        quality_report['structure_analysis'] = {
            'total_lines': total_lines,
            'valid_lines': valid_lines,
            'valid_ratio': valid_lines / total_lines if total_lines > 0 else 0,
            'unique_fields': len(all_fields),
            'avg_fields_per_line': np.mean(list(field_counts.keys())) if field_counts else 0,
            'field_distribution': field_counts
        }
        
        # Временной анализ
        if timestamps:
            timestamps = sorted(timestamps)
            time_diffs = [
                (timestamps[i+1] - timestamps[i]).total_seconds() 
                for i in range(len(timestamps)-1)
            ]
            
            quality_report['temporal_analysis'] = {
                'time_span': (timestamps[-1] - timestamps[0]).total_seconds() / 3600,  # hours
                'avg_interval': np.mean(time_diffs) if time_diffs else 0,  # seconds
                'irregular_intervals': sum(1 for diff in time_diffs if diff > 120),  # > 2 minutes
                'data_gaps': sum(1 for diff in time_diffs if diff > 300)  # > 5 minutes
            }
        
        # Рекомендации
        recommendations = []
        
        if quality_report['structure_analysis']['valid_ratio'] < 0.8:
            recommendations.append("Низкий процент валидных строк - рекомендуется очистка данных")
        
        if quality_report['temporal_analysis'].get('data_gaps', 0) > 5:
            recommendations.append("Обнаружены значительные пропуски во времени")
        
        if quality_report['structure_analysis']['unique_fields'] < 10:
            recommendations.append("Мало уникальных полей - возможны проблемы с парсингом")
        
        if not recommendations:
            recommendations.append("Качество данных хорошее")
        
        quality_report['recommendations'] = recommendations
        
        # Вывод отчета
        print("\n📋 ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ:")
        print(f"   Размер файла: {quality_report['file_info']['file_size'] / 1024:.1f} KB")
        print(f"   Валидных строк: {valid_lines}/{total_lines} ({quality_report['structure_analysis']['valid_ratio']:.1%})")
        print(f"   Уникальных полей: {quality_report['structure_analysis']['unique_fields']}")
        print(f"   Временной охват: {quality_report['temporal_analysis'].get('time_span', 0):.1f} часов")
        print("\n💡 Рекомендации:")
        for rec in recommendations:
            print(f"   - {rec}")
        
        return quality_report


def main():
    """Главная функция для запуска утилит из командной строки"""
    parser = argparse.ArgumentParser(description='Утилиты для работы с финансовыми данными')
    parser.add_argument('command', choices=[
        'validate', 'clean', 'convert', 'split', 'merge', 'sample', 'quality'
    ], help='Команда для выполнения')
    parser.add_argument('input', help='Входной файл или папка')
    parser.add_argument('-o', '--output', help='Выходной файл')
    parser.add_argument('-n', '--num-records', type=int, default=200, 
                       help='Количество записей для генерации (по умолчанию: 200)')
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    try:
        if args.command == 'validate':
            processor.validate_log_format(args.input)
            
        elif args.command == 'clean':
            processor.clean_log_data(args.input, args.output)
            
        elif args.command == 'convert':
            processor.convert_google_sheets_export(args.input, args.output)
            
        elif args.command == 'split':
            processor.split_by_events(args.input, args.output)
            
        elif args.command == 'merge':
            # Для merge входной параметр должен быть папкой
            input_path = Path(args.input)
            if input_path.is_dir():
                files = list(input_path.glob('*.txt'))
                output_file = args.output or 'merged_log.txt'
                processor.merge_log_files(files, output_file)
            else:
                print("❌ Для merge укажите папку с файлами")
                
        elif args.command == 'sample':
            output_file = args.output or 'sample_data.txt'
            processor.generate_sample_data(output_file, args.num_records)
            
        elif args.command == 'quality':
            processor.analyze_data_quality(args.input)
            
    except Exception as e:
        print(f"❌ Ошибка выполнения команды: {e}")


if __name__ == "__main__":
    main()