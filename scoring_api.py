#!/usr/bin/env python3
"""
API для интеграции скоринговой системы
Простой интерфейс для использования обученной модели
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ScoringAPI:
    """
    API для применения обученной скоринговой системы
    """
    
    def __init__(self, config_path="results/scoring_config.json", weights_path="results/weight_matrix.csv"):
        """
        Инициализация API
        
        Args:
            config_path: путь к конфигурации скоринга
            weights_path: путь к матрице весов
        """
        self.config = None
        self.weights = None
        self.thresholds = {}
        self.field_weights = {}
        self.is_ready = False
        
        try:
            self.load_configuration(config_path, weights_path)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            print("💡 Запустите сначала обучение модели")
    
    def load_configuration(self, config_path, weights_path):
        """Загрузка конфигурации и весов"""
        print("📁 Загрузка конфигурации скоринга...")
        
        # Загрузка конфигурации
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.thresholds = self.config.get('thresholds', {})
            self.field_weights = self.config.get('weights', {})
        else:
            raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")
        
        # Загрузка матрицы весов
        if Path(weights_path).exists():
            self.weights = pd.read_csv(weights_path)
        else:
            print(f"⚠️ Матрица весов не найдена: {weights_path}")
        
        self.is_ready = True
        print("✅ Конфигурация загружена")
    
    def parse_log_line(self, log_line):
        """
        Парсинг одной строки лога в структурированные данные
        
        Args:
            log_line: строка лога
            
        Returns:
            dict: распарсенные данные
        """
        if not isinstance(log_line, str) or not log_line.strip():
            return {}
        
        data = {}
        
        try:
            # Базовый парсинг
            if '|' not in log_line:
                return {}
            
            # Разделение на части
            parts = log_line.split('|')
            
            if len(parts) < 6:
                return {}
            
            # Извлечение основных данных
            try:
                data['color'] = parts[4] if len(parts) > 4 else 'UNKNOWN'
                data['price_change'] = self._parse_percentage(parts[5]) if len(parts) > 5 else 0
                data['volume'] = self._parse_volume(parts[6]) if len(parts) > 6 else 0
            except (IndexError, ValueError):
                pass
            
            # Поиск OHLC данных
            remaining_data = '|'.join(parts[6:]) if len(parts) > 6 else ''
            
            ohlc_match = re.search(r'o:([\d.]+).*?h:([\d.]+).*?l:([\d.]+).*?c:([\d.]+)', remaining_data)
            if ohlc_match:
                data['open'] = float(ohlc_match.group(1))
                data['high'] = float(ohlc_match.group(2))
                data['low'] = float(ohlc_match.group(3))
                data['close'] = float(ohlc_match.group(4))
                data['range'] = data['high'] - data['low']
            
            # Парсинг полей
            import re
            field_pattern = r'([a-zA-Z]+\d*)-?([\d.-]+%?[a-zA-Z]*!*)'
            fields = re.findall(field_pattern, remaining_data)
            
            for field_name, field_value in fields:
                if field_name in ['o', 'h', 'l', 'c', 'rng']:
                    continue
                
                try:
                    # Очистка и конвертация значения
                    clean_value = field_value.replace('%', '').replace('!', '').replace('σ', '')
                    if clean_value.replace('.', '').replace('-', '').isdigit():
                        data[field_name] = float(clean_value)
                    else:
                        data[field_name] = field_value
                except ValueError:
                    data[field_name] = field_value
            
        except Exception as e:
            print(f"⚠️ Ошибка парсинга строки: {e}")
        
        return data
    
    def _parse_percentage(self, value_str):
        """Парсинг процентных значений"""
        try:
            if '%' in value_str:
                return float(value_str.replace('%', ''))
            return float(value_str)
        except ValueError:
            return 0.0
    
    def _parse_volume(self, volume_str):
        """Парсинг объемных данных"""
        try:
            volume_str = volume_str.upper()
            if 'K' in volume_str:
                return float(volume_str.replace('K', '')) * 1000
            elif 'M' in volume_str:
                return float(volume_str.replace('M', '')) * 1000000
            return float(volume_str)
        except ValueError:
            return 0.0
    
    def calculate_score(self, data_dict):
        """
        Расчет скора для данных
        
        Args:
            data_dict: словарь с данными полей
            
        Returns:
            dict: результат скоринга
        """
        if not self.is_ready:
            return {'error': 'API не готов', 'score': 0, 'confidence': 0}
        
        try:
            score = 0.0
            active_features = 0
            feature_contributions = {}
            
            # Проход по всем известным полям
            for field_name, threshold in self.thresholds.items():
                if field_name in data_dict:
                    field_value = data_dict[field_name]
                    
                    # Проверка активации поля
                    if isinstance(field_value, (int, float)) and abs(field_value) > threshold:
                        # Поле активировано
                        feature_key = f"{field_name}_activated"
                        weight = self.field_weights.get(feature_key, 0)
                        
                        if weight > 0:
                            contribution = weight * min(abs(field_value) / threshold, 3.0)  # Ограничение влияния
                            score += contribution
                            active_features += 1
                            feature_contributions[field_name] = {
                                'value': field_value,
                                'threshold': threshold,
                                'weight': weight,
                                'contribution': contribution
                            }
            
            # Нормализация скора
            if active_features > 0:
                normalized_score = min(score / active_features, 1.0)
                confidence = min(active_features / 5.0, 1.0)  # Больше активных полей = больше уверенности
            else:
                normalized_score = 0.0
                confidence = 0.0
            
            return {
                'score': normalized_score,
                'confidence': confidence,
                'active_features': active_features,
                'feature_contributions': feature_contributions,
                'raw_score': score
            }
            
        except Exception as e:
            return {'error': str(e), 'score': 0, 'confidence': 0}
    
    def score_log_line(self, log_line):
        """
        Скоринг одной строки лога
        
        Args:
            log_line: строка лога
            
        Returns:
            dict: результат скоринга
        """
        # Парсинг строки
        data = self.parse_log_line(log_line)
        
        if not data:
            return {'error': 'Не удалось распарсить строку', 'score': 0, 'confidence': 0}
        
        # Расчет скора
        result = self.calculate_score(data)
        result['timestamp'] = datetime.now().isoformat()
        result['parsed_fields'] = len(data)
        
        return result
    
    def score_multiple_lines(self, log_lines):
        """
        Скоринг нескольких строк лога
        
        Args:
            log_lines: список строк лога
            
        Returns:
            list: результаты скоринга для каждой строки
        """
        results = []
        
        for i, line in enumerate(log_lines):
            result = self.score_log_line(line)
            result['line_number'] = i + 1
            results.append(result)
        
        return results
    
    def score_file(self, file_path, output_path=None):
        """
        Скоринг файла лога
        
        Args:
            file_path: путь к файлу лога
            output_path: путь для сохранения результатов
            
        Returns:
            dict: сводные результаты
        """
        print(f"📊 Скоринг файла: {file_path}")
        
        if not Path(file_path).exists():
            return {'error': 'Файл не найден'}
        
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                result = self.score_log_line(line)
                result['line_number'] = line_num
                results.append(result)
        
        # Сводная статистика
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            scores = [r['score'] for r in valid_results]
            confidences = [r['confidence'] for r in valid_results]
            
            summary = {
                'total_lines': len(results),
                'valid_lines': len(valid_results),
                'avg_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'avg_confidence': np.mean(confidences),
                'high_score_lines': len([s for s in scores if s > 0.7]),
                'low_score_lines': len([s for s in scores if s < 0.3])
            }
        else:
            summary = {'error': 'Нет валидных результатов'}
        
        # Сохранение результатов
        if output_path:
            output_data = {
                'summary': summary,
                'line_results': results,
                'generation_time': datetime.now().isoformat(),
                'config_used': self.config
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Результаты сохранены: {output_path}")
        
        print(f"✅ Скоринг завершен. Средний скор: {summary.get('avg_score', 0):.3f}")
        
        return {'summary': summary, 'results': results}
    
    def get_top_scoring_lines(self, results, top_n=10):
        """
        Получение строк с наивысшими скорами
        
        Args:
            results: результаты скоринга
            top_n: количество топ строк
            
        Returns:
            list: топ строки с деталями
        """
        valid_results = [r for r in results if 'error' not in r and r['score'] > 0]
        top_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)[:top_n]
        
        return top_results
    
    def get_feature_statistics(self, results):
        """
        Статистика по активации полей
        
        Args:
            results: результаты скоринга
            
        Returns:
            dict: статистика полей
        """
        field_stats = {}
        
        for result in results:
            if 'feature_contributions' in result:
                for field_name, details in result['feature_contributions'].items():
                    if field_name not in field_stats:
                        field_stats[field_name] = {
                            'activation_count': 0,
                            'total_contribution': 0,
                            'avg_value': 0,
                            'max_value': 0,
                            'values': []
                        }
                    
                    stats = field_stats[field_name]
                    stats['activation_count'] += 1
                    stats['total_contribution'] += details['contribution']
                    stats['values'].append(details['value'])
                    stats['max_value'] = max(stats['max_value'], abs(details['value']))
        
        # Расчет средних значений
        for field_name, stats in field_stats.items():
            if stats['values']:
                stats['avg_value'] = np.mean(stats['values'])
                stats['std_value'] = np.std(stats['values'])
                stats['avg_contribution'] = stats['total_contribution'] / stats['activation_count']
        
        return field_stats
    
    def create_monitoring_dashboard_data(self, file_path, window_size=50):
        """
        Создание данных для dashboard мониторинга
        
        Args:
            file_path: путь к файлу лога
            window_size: размер скользящего окна
            
        Returns:
            dict: данные для dashboard
        """
        results = self.score_file(file_path)
        
        if 'error' in results:
            return results
        
        line_results = results['results']
        valid_results = [r for r in line_results if 'error' not in r]
        
        # Скользящие средние
        scores = [r['score'] for r in valid_results]
        rolling_avg = []
        
        for i in range(len(scores)):
            start_idx = max(0, i - window_size + 1)
            window_scores = scores[start_idx:i+1]
            rolling_avg.append(np.mean(window_scores))
        
        # Обнаружение аномалий (скор > 0.8)
        anomalies = [
            {'line': r['line_number'], 'score': r['score'], 'confidence': r['confidence']}
            for r in valid_results if r['score'] > 0.8
        ]
        
        dashboard_data = {
            'timeline': {
                'scores': scores,
                'rolling_average': rolling_avg,
                'line_numbers': [r['line_number'] for r in valid_results]
            },
            'anomalies': anomalies,
            'statistics': results['summary'],
            'field_activity': self.get_feature_statistics(valid_results),
            'alert_level': 'HIGH' if len(anomalies) > 5 else 'MEDIUM' if len(anomalies) > 0 else 'LOW'
        }
        
        return dashboard_data


class RealTimeScoringAPI(ScoringAPI):
    """
    API для скоринга в реальном времени
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_scores = []
        self.alert_threshold = 0.8
        self.max_history = 1000
    
    def process_realtime_line(self, log_line):
        """
        Обработка строки в реальном времени
        
        Args:
            log_line: новая строка лога
            
        Returns:
            dict: результат с рекомендациями
        """
        result = self.score_log_line(log_line)
        
        # Добавление в историю
        self.recent_scores.append(result)
        
        # Ограничение размера истории
        if len(self.recent_scores) > self.max_history:
            self.recent_scores = self.recent_scores[-self.max_history:]
        
        # Анализ тренда
        recent_scores = [r['score'] for r in self.recent_scores[-10:] if 'error' not in r]
        
        if len(recent_scores) >= 3:
            trend = 'RISING' if recent_scores[-1] > recent_scores[-3] else 'FALLING'
            avg_recent = np.mean(recent_scores[-5:])
        else:
            trend = 'STABLE'
            avg_recent = result['score']
        
        # Генерация алертов
        alerts = []
        
        if result['score'] > self.alert_threshold:
            alerts.append({
                'type': 'HIGH_SCORE',
                'message': f"Высокий скор: {result['score']:.3f}",
                'severity': 'HIGH'
            })
        
        if trend == 'RISING' and avg_recent > 0.6:
            alerts.append({
                'type': 'RISING_TREND',
                'message': f"Возрастающий тренд: {avg_recent:.3f}",
                'severity': 'MEDIUM'
            })
        
        # Дополнение результата
        result.update({
            'trend': trend,
            'recent_average': avg_recent,
            'alerts': alerts,
            'history_size': len(self.recent_scores)
        })
        
        return result
    
    def get_current_statistics(self):
        """Получение текущей статистики"""
        if not self.recent_scores:
            return {'message': 'Нет данных'}
        
        valid_scores = [r['score'] for r in self.recent_scores if 'error' not in r]
        
        if not valid_scores:
            return {'message': 'Нет валидных скоров'}
        
        return {
            'total_processed': len(self.recent_scores),
            'avg_score': np.mean(valid_scores),
            'max_score': np.max(valid_scores),
            'recent_avg': np.mean(valid_scores[-10:]) if len(valid_scores) >= 10 else np.mean(valid_scores),
            'alert_count': len([r for r in self.recent_scores if r.get('alerts', [])]),
            'last_update': datetime.now().isoformat()
        }


def main():
    """Пример использования API"""
    print("🚀 Демонстрация Scoring API")
    print("=" * 40)
    
    # Создание API
    api = ScoringAPI()
    
    if not api.is_ready:
        print("❌ API не готов. Сначала обучите модель.")
        return
    
    # Пример строки лога
    sample_line = """[2024-08-05T09:30:00.000+03:00]: LTF|event_test|1|2024-08-05 06:30|GREEN|0.91%|2.1K|NORMAL|97%|-16.92%_24h|o:50010|h:50475.5|l:50004|c:50465.5|rng:471.5|p2-0,p5-0,p15-0,p30-0,rd5-12.3%,mo15-85,cvz2--2.1"""
    
    print("📝 Пример строки лога:")
    print(sample_line[:100] + "...")
    print()
    
    # Скоринг одной строки
    result = api.score_log_line(sample_line)
    
    print("📊 Результат скоринга:")
    print(f"   Скор: {result.get('score', 0):.3f}")
    print(f"   Уверенность: {result.get('confidence', 0):.3f}")
    print(f"   Активных полей: {result.get('active_features', 0)}")
    
    if 'feature_contributions' in result:
        print("   Вклад полей:")
        for field, contrib in list(result['feature_contributions'].items())[:3]:
            print(f"     {field}: {contrib['contribution']:.3f}")
    
    print()
    print("✅ Демонстрация завершена")


if __name__ == "__main__":
    main()