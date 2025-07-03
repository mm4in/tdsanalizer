#!/usr/bin/env python3
"""
Улучшенный анализатор событий с понятными объяснениями
Превращает технические события в практические торговые сигналы
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

class EnhancedEventsAnalyzer:
    """Улучшенный анализатор событий с практическими объяснениями"""
    
    def __init__(self):
        self.events_data = None
        self.price_data = None
        self.field_data = None
        self.practical_events = {}
        
    def load_events_data(self, events_file="results/advanced_events/advanced_events_data.csv"):
        """Загрузка данных о событиях"""
        try:
            self.events_data = pd.read_csv(events_file)
            print(f"✅ Загружено событий: {len(self.events_data)}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки событий: {e}")
            return False
    
    def analyze_practical_events(self):
        """Анализ событий с практической точки зрения"""
        if self.events_data is None:
            return None
        
        print("🎯 Анализ событий с практической точки зрения...")
        
        # Анализ каждого типа события
        event_types = [
            'retracement_2_3pct', 'retracement_3_5pct', 'retracement_5_7pct',
            'retracement_7_10pct', 'retracement_10pct_plus',
            'culmination', 'continuation', 'consolidation', 'transition_zone'
        ]
        
        for event_type in event_types:
            if event_type in self.events_data.columns:
                analysis = self._analyze_event_type(event_type)
                self.practical_events[event_type] = analysis
        
        return self.practical_events
    
    def _analyze_event_type(self, event_type):
        """Детальный анализ конкретного типа события"""
        event_mask = self.events_data[event_type] == 1
        event_indices = self.events_data[event_mask].index.tolist()
        
        if len(event_indices) == 0:
            return {
                'count': 0,
                'frequency': 0,
                'description': 'Событие не обнаружено',
                'practical_meaning': 'Нет данных для анализа'
            }
        
        # Базовая статистика
        count = len(event_indices)
        frequency = count / len(self.events_data) * 100
        
        # Анализ контекста события
        context_analysis = self._analyze_event_context(event_indices, event_type)
        
        # Временной анализ
        timing_analysis = self._analyze_event_timing(event_indices)
        
        # Ценовой анализ
        price_analysis = self._analyze_price_movements(event_indices)
        
        # Практическое значение
        practical_meaning = self._get_practical_meaning(event_type, context_analysis, price_analysis)
        
        return {
            'count': count,
            'frequency': frequency,
            'description': self._get_event_description(event_type),
            'practical_meaning': practical_meaning,
            'context': context_analysis,
            'timing': timing_analysis,
            'price_movements': price_analysis,
            'trading_signals': self._generate_trading_signals(event_type, context_analysis)
        }
    
    def _analyze_event_context(self, event_indices, event_type):
        """Анализ контекста события (поля, активные в момент события)"""
        if len(event_indices) == 0:
            return {}
        
        # Получаем данные в моменты событий
        event_data = self.events_data.iloc[event_indices]
        
        # Анализируем активные поля
        active_fields = {}
        field_columns = [col for col in self.events_data.columns 
                        if col not in ['open', 'high', 'low', 'close', 'volume', 'range', 
                                     'price_change', 'completion', 'movement_24h'] 
                        and not col.startswith('retracement_') 
                        and col not in ['culmination', 'continuation', 'consolidation', 'transition_zone']]
        
        for field in field_columns:
            if field in event_data.columns:
                field_values = pd.to_numeric(event_data[field], errors='coerce').fillna(0)
                non_zero_values = field_values[field_values != 0]
                
                if len(non_zero_values) > 0:
                    active_fields[field] = {
                        'activation_rate': len(non_zero_values) / len(event_indices),
                        'avg_value': non_zero_values.mean(),
                        'max_value': non_zero_values.max(),
                        'min_value': non_zero_values.min()
                    }
        
        # Сортируем поля по частоте активации
        sorted_fields = sorted(active_fields.items(), 
                             key=lambda x: x[1]['activation_rate'], 
                             reverse=True)
        
        return {
            'most_active_fields': dict(sorted_fields[:10]),
            'total_active_fields': len(active_fields),
            'avg_field_activity': np.mean([info['activation_rate'] for info in active_fields.values()]) if active_fields else 0
        }
    
    def _analyze_event_timing(self, event_indices):
        """Анализ временных характеристик событий"""
        if len(event_indices) < 2:
            return {}
        
        # Интервалы между событиями
        intervals = []
        for i in range(1, len(event_indices)):
            interval = event_indices[i] - event_indices[i-1]
            intervals.append(interval)
        
        if intervals:
            return {
                'avg_interval': np.mean(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals),
                'std_interval': np.std(intervals),
                'typical_duration': f"{np.mean(intervals):.1f} периодов"
            }
        
        return {}
    
    def _analyze_price_movements(self, event_indices):
        """Анализ ценовых движений во время событий"""
        if len(event_indices) == 0:
            return {}
        
        event_data = self.events_data.iloc[event_indices]
        
        # Проверяем наличие ценовых данных
        price_columns = ['open', 'high', 'low', 'close', 'price_change']
        available_columns = [col for col in price_columns if col in event_data.columns]
        
        if not available_columns:
            return {'error': 'Ценовые данные недоступны'}
        
        analysis = {}
        
        # Анализ изменений цены
        if 'price_change' in event_data.columns:
            price_changes = pd.to_numeric(event_data['price_change'], errors='coerce').fillna(0)
            analysis['price_change'] = {
                'avg': price_changes.mean(),
                'positive_events': (price_changes > 0).sum(),
                'negative_events': (price_changes < 0).sum(),
                'neutral_events': (price_changes == 0).sum(),
                'max_gain': price_changes.max(),
                'max_loss': price_changes.min()
            }
        
        # Анализ волатильности
        if all(col in event_data.columns for col in ['high', 'low', 'close']):
            highs = pd.to_numeric(event_data['high'], errors='coerce').fillna(0)
            lows = pd.to_numeric(event_data['low'], errors='coerce').fillna(0)
            closes = pd.to_numeric(event_data['close'], errors='coerce').fillna(0)
            
            # Расчет диапазонов
            ranges = ((highs - lows) / closes * 100).replace([np.inf, -np.inf], 0)
            
            analysis['volatility'] = {
                'avg_range_pct': ranges.mean(),
                'max_range_pct': ranges.max(),
                'high_volatility_events': (ranges > ranges.quantile(0.8)).sum()
            }
        
        return analysis
    
    def _get_event_description(self, event_type):
        """Получение описания события"""
        descriptions = {
            'retracement_2_3pct': 'Небольшой откат 2-3% - техническая коррекция',
            'retracement_3_5pct': 'Умеренный откат 3-5% - возможность входа',
            'retracement_5_7pct': 'Значительный откат 5-7% - сильная коррекция',
            'retracement_7_10pct': 'Глубокий откат 7-10% - тестирование поддержки',
            'retracement_10pct_plus': 'Экстремальный откат 10%+ - возможный разворот',
            'culmination': 'Кульминация - точка разворота тренда',
            'continuation': 'Продолжение - пробой и развитие движения',
            'consolidation': 'Консолидация - боковое движение, накопление',
            'transition_zone': 'Переходная зона - неопределенность направления'
        }
        
        return descriptions.get(event_type, 'Неизвестный тип события')
    
    def _get_practical_meaning(self, event_type, context, price_analysis):
        """Получение практического значения события"""
        meanings = {
            'retracement_2_3pct': 'Хорошая точка для добавления к позиции по тренду',
            'retracement_3_5pct': 'Классическая коррекция - возможность входа',
            'retracement_5_7pct': 'Глубокая коррекция - входить с осторожностью',
            'retracement_7_10pct': 'Возможное изменение тренда - требует подтверждения',
            'retracement_10pct_plus': 'Высокая вероятность разворота тренда',
            'culmination': 'Ожидать смену направления движения',
            'continuation': 'Подтверждение текущего тренда - можно следовать',
            'consolidation': 'Ожидание пробоя - готовиться к движению',
            'transition_zone': 'Неопределенность - лучше остаться в стороне'
        }
        
        base_meaning = meanings.get(event_type, 'Требует дополнительного анализа')
        
        # Добавляем контекстную информацию
        if context and 'most_active_fields' in context:
            top_fields = list(context['most_active_fields'].keys())[:3]
            if top_fields:
                base_meaning += f" (активны: {', '.join(top_fields)})"
        
        return base_meaning
    
    def _generate_trading_signals(self, event_type, context):
        """Генерация торговых сигналов для события"""
        signals = {
            'retracement_2_3pct': {
                'action': 'BUY_DIP',
                'confidence': 'HIGH',
                'risk': 'LOW',
                'comment': 'Добавление к позиции по тренду'
            },
            'retracement_3_5pct': {
                'action': 'BUY_PULLBACK',
                'confidence': 'HIGH',
                'risk': 'MEDIUM',
                'comment': 'Классический вход после коррекции'
            },
            'retracement_5_7pct': {
                'action': 'WAIT_CONFIRMATION',
                'confidence': 'MEDIUM',
                'risk': 'MEDIUM',
                'comment': 'Ждать подтверждения продолжения тренда'
            },
            'retracement_7_10pct': {
                'action': 'CAUTIOUS_ENTRY',
                'confidence': 'LOW',
                'risk': 'HIGH',
                'comment': 'Возможен разворот - малый размер позиции'
            },
            'retracement_10pct_plus': {
                'action': 'REVERSE_SIGNAL',
                'confidence': 'MEDIUM',
                'risk': 'HIGH',
                'comment': 'Рассмотреть смену направления'
            },
            'culmination': {
                'action': 'PREPARE_REVERSE',
                'confidence': 'HIGH',
                'risk': 'MEDIUM',
                'comment': 'Готовиться к развороту тренда'
            },
            'continuation': {
                'action': 'FOLLOW_TREND',
                'confidence': 'HIGH',
                'risk': 'LOW',
                'comment': 'Следовать пробою в направлении тренда'
            },
            'consolidation': {
                'action': 'WAIT_BREAKOUT',
                'confidence': 'MEDIUM',
                'risk': 'MEDIUM',
                'comment': 'Ожидать пробоя границ консолидации'
            },
            'transition_zone': {
                'action': 'STAY_ASIDE',
                'confidence': 'LOW',
                'risk': 'HIGH',
                'comment': 'Неопределенность - лучше наблюдать'
            }
        }
        
        return signals.get(event_type, {
            'action': 'ANALYZE_FURTHER',
            'confidence': 'LOW',
            'risk': 'UNKNOWN',
            'comment': 'Требует дополнительного анализа'
        })
    
    def create_practical_events_report(self):
        """Создание практического отчета по событиям"""
        if not self.practical_events:
            return None
        
        report_lines = [
            "📈 ПРАКТИЧЕСКИЙ АНАЛИЗ РЫНОЧНЫХ СОБЫТИЙ",
            "=" * 60,
            f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "🎯 КРАТКАЯ СВОДКА:"
        ]
        
        # Сводка по событиям
        total_events = sum(event['count'] for event in self.practical_events.values())
        report_lines.extend([
            f"   Всего событий проанализировано: {total_events}",
            f"   Типов событий: {len(self.practical_events)}",
            ""
        ])
        
        # Детальный анализ каждого события
        for event_type, analysis in self.practical_events.items():
            if analysis['count'] == 0:
                continue
                
            report_lines.extend([
                f"📊 {analysis['description'].upper()}",
                "-" * 40,
                f"Количество: {analysis['count']} ({analysis['frequency']:.1f}%)",
                f"Практическое значение: {analysis['practical_meaning']}",
            ])
            
            # Торговые сигналы
            if 'trading_signals' in analysis:
                signals = analysis['trading_signals']
                report_lines.extend([
                    f"Торговый сигнал: {signals['action']}",
                    f"Уверенность: {signals['confidence']}",
                    f"Риск: {signals['risk']}",
                    f"Комментарий: {signals['comment']}"
                ])
            
            # Активные поля
            if 'context' in analysis and 'most_active_fields' in analysis['context']:
                active_fields = analysis['context']['most_active_fields']
                if active_fields:
                    top_3_fields = list(active_fields.keys())[:3]
                    report_lines.append(f"Ключевые индикаторы: {', '.join(top_3_fields)}")
            
            # Ценовые характеристики
            if 'price_movements' in analysis and 'price_change' in analysis['price_movements']:
                price_data = analysis['price_movements']['price_change']
                report_lines.extend([
                    f"Среднее изменение цены: {price_data['avg']:.2f}%",
                    f"Положительных событий: {price_data['positive_events']}",
                    f"Отрицательных событий: {price_data['negative_events']}"
                ])
            
            report_lines.append("")
        
        # Общие рекомендации
        report_lines.extend([
            "💡 ОБЩИЕ РЕКОМЕНДАЦИИ:",
            "1. Откаты 2-5% - хорошие точки входа по тренду",
            "2. Откаты 7%+ - повышенная осторожность, возможен разворот",
            "3. Консолидации - ожидание пробоя для определения направления", 
            "4. Кульминации - подготовка к смене тренда",
            "5. Продолжения - подтверждение текущего направления",
            "",
            "⚠️ ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ:",
            "- Используйте события как дополнение к основному анализу",
            "- Всегда ставьте стоп-лоссы",
            "- Учитывайте общий рыночный контекст",
            "- Тестируйте стратегии на исторических данных",
            "",
            "=" * 60
        ])
        
        # Сохранение отчета
        output_file = Path("results") / "ПРАКТИЧЕСКИЙ_АНАЛИЗ_СОБЫТИЙ.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Практический отчет создан: {output_file}")
        return '\n'.join(report_lines)
    
    def create_events_summary_table(self):
        """Создание сводной таблицы событий"""
        if not self.practical_events:
            return None
        
        summary_data = []
        
        for event_type, analysis in self.practical_events.items():
            if analysis['count'] == 0:
                continue
            
            signals = analysis.get('trading_signals', {})
            
            summary_data.append({
                'Событие': analysis['description'],
                'Количество': analysis['count'],
                'Частота_%': f"{analysis['frequency']:.1f}%",
                'Торговый_сигнал': signals.get('action', 'N/A'),
                'Уверенность': signals.get('confidence', 'N/A'),
                'Риск': signals.get('risk', 'N/A'),
                'Практическое_значение': analysis['practical_meaning'][:50] + '...' if len(analysis['practical_meaning']) > 50 else analysis['practical_meaning']
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Сохранение таблицы
            output_file = Path("results") / "events_summary_table.csv"
            summary_df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"📊 Сводная таблица создана: {output_file}")
            return summary_df
        
        return None


def main():
    """Запуск улучшенного анализа событий"""
    analyzer = EnhancedEventsAnalyzer()
    
    # Загрузка данных
    if not analyzer.load_events_data():
        print("❌ Не удалось загрузить данные событий")
        return
    
    # Анализ событий
    practical_events = analyzer.analyze_practical_events()
    
    if practical_events:
        print("✅ Практический анализ событий завершен")
        
        # Создание отчетов
        analyzer.create_practical_events_report()
        analyzer.create_events_summary_table()
        
        print("\n📋 Создано:")
        print("   - ПРАКТИЧЕСКИЙ_АНАЛИЗ_СОБЫТИЙ.txt")
        print("   - events_summary_table.csv")
    else:
        print("❌ Не удалось выполнить анализ событий")


if __name__ == "__main__":
    main()