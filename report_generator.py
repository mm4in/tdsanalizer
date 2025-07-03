#!/usr/bin/env python3
"""
Генератор ПОНЯТНЫХ отчетов для трейдеров
Превращает технические результаты в практические выводы
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class ClearReportGenerator:
    """Генератор понятных отчетов из технических результатов"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.weight_matrix = None
        self.scoring_config = None
        self.temporal_lags = None
        self.veto_rules = None
        
    def load_results(self):
        """Загрузка всех результатов анализа"""
        try:
            # Основные результаты
            if (self.results_dir / "weight_matrix.csv").exists():
                self.weight_matrix = pd.read_csv(self.results_dir / "weight_matrix.csv")
            
            if (self.results_dir / "scoring_config.json").exists():
                with open(self.results_dir / "scoring_config.json", 'r') as f:
                    self.scoring_config = json.load(f)
            
            # LTF результаты
            if (self.results_dir / "ltf" / "temporal_lags_ltf.csv").exists():
                self.temporal_lags = pd.read_csv(self.results_dir / "ltf" / "temporal_lags_ltf.csv")
            
            # VETO результаты
            if (self.results_dir / "veto_system" / "veto_rules.json").exists():
                with open(self.results_dir / "veto_system" / "veto_rules.json", 'r') as f:
                    self.veto_rules = json.load(f)
                    
            return True
        except Exception as e:
            print(f"Ошибка загрузки результатов: {e}")
            return False
    
    def generate_trader_friendly_report(self):
        """Создание отчета понятного для трейдера"""
        if not self.load_results():
            return None
        
        report_lines = [
            "🎯 ФИНАНСОВЫЙ АНАЛИЗАТОР - ПРАКТИЧЕСКИЕ ВЫВОДЫ",
            "=" * 60,
            f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "🚀 КРАТКАЯ СВОДКА:",
        ]
        
        # Основные метрики
        if self.scoring_config:
            validation_score = self.scoring_config.get('validation_score', 0)
            report_lines.extend([
                f"✅ Точность системы: {validation_score:.1%}",
                f"✅ Качество сигналов: {'ОТЛИЧНОЕ' if validation_score > 0.8 else 'ХОРОШЕЕ' if validation_score > 0.6 else 'СРЕДНЕЕ'}",
                f"✅ Найдено полезных полей: {len(self.scoring_config.get('thresholds', {}))}"
            ])
        
        report_lines.extend([
            "",
            "💎 ТОП-10 САМЫХ ВАЖНЫХ ПОЛЕЙ (СИГНАЛОВ):",
            "   (чем выше вес, тем важнее поле для прогноза)"
        ])
        
        # Топ полей по важности
        top_fields = self._get_top_fields()
        for i, (field, weight, description) in enumerate(top_fields[:10], 1):
            report_lines.append(f"   {i:2d}. {field:15s} (вес: {weight:.3f}) - {description}")
        
        report_lines.extend([
            "",
            "⚡ ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ СИГНАЛОВ:",
        ])
        
        # Временные лаги
        timing_info = self._analyze_timing()
        for group, info in timing_info.items():
            report_lines.append(f"   {group}: срабатывает за {info['lag']:.1f} периодов, надежность {info['reliability']}")
        
        report_lines.extend([
            "",
            "🛡️ СТОП-СИГНАЛЫ (когда НЕ входить в сделку):",
        ])
        
        # Стоп-поля
        stop_signals = self._get_stop_signals()
        for field, reason in stop_signals[:5]:
            report_lines.append(f"   ❌ {field}: {reason}")
        
        report_lines.extend([
            "",
            "📊 ТИПЫ РЫНОЧНЫХ СОБЫТИЙ:",
        ])
        
        # События
        events_info = self._analyze_events()
        for event_type, info in events_info.items():
            report_lines.append(f"   📈 {event_type}: {info['description']} ({info['frequency']})")
        
        report_lines.extend([
            "",
            "💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:",
        ])
        
        # Рекомендации
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"   {i}. {rec}")
        
        report_lines.extend([
            "",
            "⚠️ ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ:",
        ])
        
        # Предупреждения
        warnings = self._generate_warnings()
        for warning in warnings:
            report_lines.append(f"   ⚠️ {warning}")
        
        report_lines.extend([
            "",
            "🔍 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ В ФАЙЛАХ:",
            "   - weight_matrix.csv - веса всех полей",
            "   - scoring_config.json - пороги активации",
            "   - results/ltf/ - анализ быстрых сигналов",
            "   - results/veto_system/ - правила блокировки",
            "",
            "=" * 60
        ])
        
        # Сохранение отчета
        output_file = self.results_dir / "ПОНЯТНЫЙ_ОТЧЕТ.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Понятный отчет создан: {output_file}")
        return '\n'.join(report_lines)
    
    def _get_top_fields(self):
        """Получение топ полей с описаниями"""
        if not self.scoring_config or 'weights' not in self.scoring_config:
            return []
        
        weights = self.scoring_config['weights']
        
        # Сортировка по важности
        sorted_fields = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for field, weight in sorted_fields:
            # Упрощение названия поля
            clean_name = field.replace('_activated', '').replace('_lag_', '_L')
            
            # Описание поля
            description = self._describe_field(clean_name)
            
            result.append((clean_name, weight, description))
        
        return result
    
    def _describe_field(self, field_name):
        """Описание поля понятным языком"""
        descriptions = {
            'volume': 'Объем торгов (активность рынка)',
            'price_change': 'Изменение цены (волатильность)',
            'co': 'Индикатор перепроданности/перекупленности',
            'mo': 'Momentum индикатор (сила движения)',
            'ro': 'Индикатор разворота',
            'as': 'Индикатор ускорения',
            'ze': 'Z-score экстремум',
            'ef': 'Фактор эффективности',
            'mz': 'Momentum Z-score',
            'rz': 'Разворот Z-score',
            'maz': 'MA Z-score',
            'cvz': 'Волатильность Z-score',
            'rd': 'Индикатор направления',
            'md': 'MA дивергенция',
            'do': 'Перекупленность',
            'so': 'Перепроданность'
        }
        
        # Поиск базового названия
        for base, desc in descriptions.items():
            if field_name.startswith(base):
                # Добавление временного фрейма
                if any(tf in field_name for tf in ['2', '5', '15', '30']):
                    desc += ' (быстрый сигнал)'
                elif any(tf in field_name for tf in ['1h', '4h', '1d']):
                    desc += ' (медленный сигнал)'
                elif '_L' in field_name:
                    desc += ' (с задержкой)'
                return desc
        
        return 'Технический индикатор'
    
    def _analyze_timing(self):
        """Анализ временных характеристик"""
        if not self.temporal_lags:
            return {}
        
        timing = {}
        
        for _, row in self.temporal_lags.iterrows():
            group = row.iloc[0]  # Первая колонка - название группы
            mean_lag = row['mean_lag']
            activation_rate = row['activation_rate']
            
            # Оценка надежности
            if activation_rate > 0.8:
                reliability = "очень высокая"
            elif activation_rate > 0.6:
                reliability = "высокая"
            elif activation_rate > 0.4:
                reliability = "средняя"
            else:
                reliability = "низкая"
            
            timing[group] = {
                'lag': mean_lag,
                'reliability': reliability,
                'activation_rate': activation_rate
            }
        
        return timing
    
    def _get_stop_signals(self):
        """Получение стоп-сигналов"""
        if not self.veto_rules:
            return []
        
        stop_signals = []
        
        # Блокирующие поля
        blocking_fields = self.veto_rules.get('blocking_fields', {})
        for field, info in blocking_fields.items():
            strength = info.get('blocking_strength', 0)
            reason = f"блокирует сигналы с силой {strength:.1%}"
            stop_signals.append((field, reason))
        
        # Ложные сигналы
        false_signals = self.veto_rules.get('false_signal_filters', {})
        for field, info in false_signals.items():
            false_rate = info.get('false_positive_rate', 0)
            reason = f"дает ложные сигналы в {false_rate:.1%} случаев"
            stop_signals.append((field, reason))
        
        # Сортировка по важности блокировки
        stop_signals.sort(key=lambda x: x[1], reverse=True)
        
        return stop_signals
    
    def _analyze_events(self):
        """Анализ типов событий"""
        # Загрузка статистики событий
        events_file = self.results_dir / "advanced_events" / "event_statistics.json"
        
        if not events_file.exists():
            return {}
        
        try:
            with open(events_file, 'r') as f:
                events_stats = json.load(f)
        except:
            return {}
        
        events_info = {}
        
        # Описание событий
        event_descriptions = {
            'retracement_2_3pct': {
                'description': 'Откат 2-3% - коррекция без обновления экстремума',
                'practical': 'Возможность входа по тренду'
            },
            'retracement_5_7pct': {
                'description': 'Откат 5-7% - значительная коррекция', 
                'practical': 'Хорошая точка входа в тренд'
            },
            'consolidation': {
                'description': 'Консолидация - боковое движение',
                'practical': 'Ожидание пробоя, осторожность'
            },
            'continuation': {
                'description': 'Продолжение движения - пробой уровней',
                'practical': 'Подтверждение направления тренда'
            },
            'culmination': {
                'description': 'Кульминация - точка разворота тренда',
                'practical': 'Возможная смена направления'
            },
            'transition_zone': {
                'description': 'Переходная зона - неопределенность',
                'practical': 'Ожидание четких сигналов'
            }
        }
        
        for event_type, stats in events_stats.items():
            if event_type == 'summary':
                continue
            
            frequency = stats.get('percentage', 0)
            
            if event_type in event_descriptions:
                info = event_descriptions[event_type]
                events_info[event_type] = {
                    'description': info['description'],
                    'frequency': f"{frequency:.1f}% случаев",
                    'practical': info['practical']
                }
        
        return events_info
    
    def _generate_recommendations(self):
        """Генерация практических рекомендаций"""
        recommendations = []
        
        # На основе результатов
        if self.scoring_config:
            validation_score = self.scoring_config.get('validation_score', 0)
            
            if validation_score > 0.9:
                recommendations.append("Система показывает отличные результаты - можно использовать для торговли")
            elif validation_score > 0.7:
                recommendations.append("Система работает хорошо - рекомендуется дополнительное тестирование")
            else:
                recommendations.append("Система требует доработки перед использованием")
        
        # Рекомендации по полям
        if self.weight_matrix is not None:
            recommendations.append("Сосредоточьтесь на топ-10 полях - они дают 80% точности")
        
        # Временные рекомендации
        if self.temporal_lags is not None:
            recommendations.append("Учитывайте временные лаги - сигналы срабатывают с задержкой")
        
        # VETO рекомендации
        if self.veto_rules:
            recommendations.append("Обязательно используйте стоп-сигналы для фильтрации ложных входов")
        
        # Общие рекомендации
        recommendations.extend([
            "Тестируйте систему на исторических данных перед реальным использованием",
            "Ведите статистику срабатываний для корректировки весов",
            "Используйте систему как дополнение к анализу, а не замену"
        ])
        
        return recommendations
    
    def _generate_warnings(self):
        """Генерация предупреждений"""
        warnings = []
        
        # HTF предупреждение
        warnings.append("HTF данные отсутствуют - анализ только по быстрым сигналам")
        
        # VETO предупреждение
        if self.veto_rules:
            validation = self.veto_rules.get('validation', {})
            veto_effectiveness = validation.get('veto_effectiveness', 0)
            
            if veto_effectiveness < 0.3:
                warnings.append("VETO система работает неэффективно - требует настройки")
        
        # Общие предупреждения
        warnings.extend([
            "Система обучена на ограниченном наборе данных",
            "Результаты могут отличаться на других временных периодах",
            "Не используйте систему без дополнительного анализа рынка"
        ])
        
        return warnings


def main():
    """Запуск генератора понятных отчетов"""
    generator = ClearReportGenerator()
    report = generator.generate_trader_friendly_report()
    
    if report:
        print("📋 ПОНЯТНЫЙ ОТЧЕТ СОЗДАН!")
        print("\n" + "="*60)
        print(report[:1000] + "..." if len(report) > 1000 else report)
        print("="*60)
    else:
        print("❌ Ошибка создания отчета")


if __name__ == "__main__":
    main()