#!/usr/bin/env python3
"""
Визуализация результатов анализа таймзон.
Создает графики и отчеты для сравнения различных интерпретаций.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any
import numpy as np


class TimezoneVisualizer:
    """Визуализатор результатов анализа таймзон"""
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
    
    def create_summary_plot(self, output_file: str = '/home/claude/timezone_summary.png'):
        """Создает сводный график по всем таймзонам"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Подготовка данных
        timezones = [r['timezone_label'] for r in self.results]
        record_counts = [r['records_count'] for r in self.results]
        state_changes = [r['state_changes_count'] for r in self.results]
        
        # График 1: Количество записей
        ax1.bar(timezones, record_counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Таймзона', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Количество записей 2704', fontsize=12, fontweight='bold')
        ax1.set_title('Количество записей по таймзонам', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Добавляем значения на столбцах
        for i, (tz, count) in enumerate(zip(timezones, record_counts)):
            if count > 0:
                ax1.text(i, count + max(record_counts) * 0.02, str(count), 
                        ha='center', va='bottom', fontweight='bold')
        
        # График 2: Количество переключений состояния
        colors = ['green' if sc > 0 else 'lightgray' for sc in state_changes]
        ax2.bar(timezones, state_changes, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Таймзона', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Количество переключений состояния', fontsize=12, fontweight='bold')
        ax2.set_title('Переключения состояния EMPTY ↔ FULL по таймзонам', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Добавляем значения на столбцах
        for i, (tz, count) in enumerate(zip(timezones, state_changes)):
            if count > 0:
                ax2.text(i, count + max(state_changes + [1]) * 0.02, str(count), 
                        ha='center', va='bottom', fontweight='bold', color='darkgreen')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {output_file}")
        plt.close()
    
    def create_detailed_report(self, output_file: str = '/home/claude/timezone_report.txt'):
        """Создает детальный текстовый отчет"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ПО АНАЛИЗУ ТАЙМЗОН CAN-ЛОГОВ\n")
            f.write("=" * 100 + "\n\n")
            
            # Общая информация
            f.write("ЦЕЛЕВОЙ ПЕРИОД (UTC+7):\n")
            if self.results:
                target = self.results[0]['target_period']
                f.write(f"  От: {target['start']}\n")
                f.write(f"  До: {target['end']}\n\n")
            
            # Статистика по каждой таймзоне
            for result in self.results:
                f.write("\n" + "─" * 100 + "\n")
                f.write(f"ТАЙМЗОНА: {result['timezone_label']}\n")
                f.write("─" * 100 + "\n\n")
                
                f.write(f"Скорректированный период ({result['timezone_label']}):\n")
                f.write(f"  От: {result['adjusted_period']['start']}\n")
                f.write(f"  До: {result['adjusted_period']['end']}\n\n")
                
                f.write(f"Найдено записей frame='2704': {result['records_count']}\n")
                
                if result['first_record']:
                    f.write(f"\nПервая запись:\n")
                    f.write(f"  Время: {result['first_record']['timestamp']}\n")
                    f.write(f"  Raw timestamp: {result['first_record']['raw_time']}\n")
                    
                    f.write(f"\nПоследняя запись:\n")
                    f.write(f"  Время: {result['last_record']['timestamp']}\n")
                    f.write(f"  Raw timestamp: {result['last_record']['raw_time']}\n")
                else:
                    f.write("\n⚠ В этом периоде нет записей\n")
                
                f.write(f"\nПереключений состояния: {result['state_changes_count']}\n")
                
                if result['state_changes']:
                    f.write("\nДЕТАЛИ ПЕРЕКЛЮЧЕНИЙ:\n")
                    f.write("-" * 80 + "\n")
                    for i, change in enumerate(result['state_changes'], 1):
                        f.write(f"{i}. {change['timestamp']} → {change['new_state']} (значение={change['value']})\n")
                    f.write("-" * 80 + "\n")
                
                f.write("\n")
            
            # Выводы и рекомендации
            f.write("\n" + "=" * 100 + "\n")
            f.write("ВЫВОДЫ И РЕКОМЕНДАЦИИ\n")
            f.write("=" * 100 + "\n\n")
            
            # Находим таймзоны с максимальным количеством записей
            max_records = max((r['records_count'] for r in self.results), default=0)
            best_tz_records = [r for r in self.results if r['records_count'] == max_records and max_records > 0]
            
            if best_tz_records:
                f.write(f"Таймзоны с максимальным количеством записей ({max_records}):\n")
                for r in best_tz_records:
                    f.write(f"  • {r['timezone_label']}\n")
                f.write("\n")
            
            # Находим таймзоны с переключениями состояния
            tz_with_changes = [r for r in self.results if r['state_changes_count'] > 0]
            
            if tz_with_changes:
                f.write(f"Таймзоны с обнаруженными переключениями состояния:\n")
                for r in tz_with_changes:
                    f.write(f"  • {r['timezone_label']}: {r['state_changes_count']} переключений\n")
                f.write("\n")
                
                f.write("РЕКОМЕНДАЦИЯ: Наиболее вероятная таймзона - та, где обнаружены переключения,\n")
                f.write("соответствующие ожидаемым событиям.\n")
            else:
                f.write("⚠ Переключения состояния не обнаружены ни в одной таймзоне.\n")
                f.write("Возможные причины:\n")
                f.write("  1. Неверный целевой период\n")
                f.write("  2. События не произошли в указанное время\n")
                f.write("  3. Данные в логах неполные\n")
                f.write("  4. Требуется корректировка THRESHOLD или CONSECUTIVE\n")
        
        print(f"✓ Отчет сохранен: {output_file}")
    
    def create_timeline_comparison(self, output_file: str = '/home/claude/timezone_timeline.png'):
        """Создает график сравнения временных линий"""
        
        # Фильтруем таймзоны с записями
        results_with_data = [r for r in self.results if r['records_count'] > 0]
        
        if not results_with_data:
            print("⚠ Нет данных для построения графика временных линий")
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for i, result in enumerate(results_with_data):
            tz_label = result['timezone_label']
            
            # Парсим времена
            start_str = result['adjusted_period']['start']
            end_str = result['adjusted_period']['end']
            
            start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
            
            # Рисуем линию периода
            ax.plot([start_dt, end_dt], [i, i], 'o-', linewidth=3, markersize=8, 
                   label=f"{tz_label} ({result['records_count']} записей)")
            
            # Отмечаем переключения состояния
            for change in result['state_changes']:
                change_dt = datetime.strptime(change['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                marker = '^' if change['new_state'] == 'FULL' else 'v'
                color = 'green' if change['new_state'] == 'FULL' else 'red'
                ax.scatter(change_dt, i, marker=marker, s=200, c=color, 
                          edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_yticks(range(len(results_with_data)))
        ax.set_yticklabels([r['timezone_label'] for r in results_with_data])
        ax.set_xlabel('Время', fontsize=12, fontweight='bold')
        ax.set_ylabel('Таймзона', fontsize=12, fontweight='bold')
        ax.set_title('Сравнение временных периодов по таймзонам\n(▲ = переход в FULL, ▼ = переход в EMPTY)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ График временных линий сохранен: {output_file}")
        plt.close()


def main():
    """Основная функция для визуализации"""
    
    print("=== Визуализация результатов анализа таймзон ===\n")
    
    # Загрузка результатов
    results_file = "/home/claude/timezone_analysis_results.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✓ Загружено результатов: {len(results)}\n")
        
        # Создание визуализаций
        visualizer = TimezoneVisualizer(results)
        
        print("Создание графиков...")
        visualizer.create_summary_plot()
        visualizer.create_timeline_comparison()
        visualizer.create_detailed_report()
        
        print("\n✓ Визуализация завершена!")
        
    except FileNotFoundError:
        print(f"✗ Файл {results_file} не найден")
        print("Сначала запустите can_log_analyzer.py для создания результатов")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
