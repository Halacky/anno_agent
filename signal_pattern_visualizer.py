#!/usr/bin/env python3
"""
Визуализация паттерна изменения сигнала frame 2704 для каждой таймзоны.
Позволяет визуально сравнить временные паттерны и определить правильную таймзону.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
import numpy as np


class SignalPatternVisualizer:
    """Визуализатор паттернов сигнала по таймзонам"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.records_2704 = []
        self._load_data()
    
    def _load_data(self):
        """Загружает данные из файла логов"""
        print(f"Загрузка данных из {self.log_file}...")
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = eval(line)
                    if record['frame_name'] == '2704':
                        self.records_2704.append(record)
                except:
                    continue
        print(f"✓ Загружено {len(self.records_2704)} записей frame='2704'")
    
    def _extract_uint16(self, data_str: str) -> int:
        """Извлекает uint16 из hex-строки данных"""
        hex_values = data_str.strip().split()
        if len(hex_values) < 7:
            return 0
        byte5 = int(hex_values[5], 16)
        byte6 = int(hex_values[6], 16)
        return byte5 + (byte6 << 8)
    
    def create_signal_plots(
        self,
        target_start: datetime,
        target_end: datetime,
        tz_range: tuple = (-7, 7),
        output_file: str = '/home/claude/signal_patterns_all.png'
    ):
        """
        Создает сетку графиков с паттернами сигнала для всех таймзон.
        """
        timezones = list(range(tz_range[0], tz_range[1] + 1))
        n_timezones = len(timezones)
        
        # Создаем сетку графиков (5 колонок)
        n_cols = 5
        n_rows = (n_timezones + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = axes.flatten() if n_timezones > 1 else [axes]
        
        # Целевой период в UTC+7
        tz_diff_base = 7
        
        for idx, tz_offset in enumerate(timezones):
            ax = axes[idx]
            
            # Корректируем целевой период для текущей таймзоны
            tz_diff = tz_offset - tz_diff_base
            adjusted_start = target_start - timedelta(hours=tz_diff)
            adjusted_end = target_end - timedelta(hours=tz_diff)
            
            # Фильтруем записи для этой таймзоны
            timestamps = []
            values = []
            
            for record in self.records_2704:
                ts = datetime.fromtimestamp(record['action_time'] / 1000.0)
                ts_adjusted = ts + timedelta(hours=tz_offset)
                
                if adjusted_start <= ts_adjusted <= adjusted_end:
                    timestamps.append(ts_adjusted)
                    values.append(self._extract_uint16(record['data']))
            
            # Построение графика
            if timestamps:
                ax.plot(timestamps, values, '-', linewidth=0.8, alpha=0.7, color='steelblue')
                ax.axhline(y=1100, color='red', linestyle='--', linewidth=1.5, 
                          label='Threshold=1100', alpha=0.7)
                ax.fill_between(timestamps, 0, values, alpha=0.3, color='steelblue')
                
                ax.set_title(f'UTC{tz_offset:+d} ({len(timestamps)} записей)', 
                           fontweight='bold', fontsize=10)
                ax.set_ylabel('Значение uint16', fontsize=8)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='both', labelsize=7)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Подсветка зон
                ax.axhspan(0, 1100, alpha=0.1, color='green', label='EMPTY zone')
                ax.axhspan(1100, max(values + [2500]), alpha=0.1, color='orange', 
                          label='FULL zone')
                
                if idx == 0:
                    ax.legend(fontsize=7, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'Нет данных\nв этом периоде', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_title(f'UTC{tz_offset:+d} (0 записей)', 
                           fontweight='bold', fontsize=10, color='gray')
                ax.set_xlim([adjusted_start, adjusted_end])
                ax.set_ylim([0, 2500])
        
        # Скрываем лишние подграфики
        for idx in range(n_timezones, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(
            f'Паттерны сигнала frame 2704 по таймзонам\n'
            f'Целевой период (UTC+7): {target_start.strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{target_end.strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Графики сохранены: {output_file}")
        plt.close()
    
    def create_detailed_signal_plot(
        self,
        timezone_offset: int,
        target_start: datetime,
        target_end: datetime,
        output_file: str = None
    ):
        """
        Создает детальный график сигнала для конкретной таймзоны.
        """
        if output_file is None:
            tz_label = f"UTC{timezone_offset:+d}".replace('+', 'plus').replace('-', 'minus')
            output_file = f'/home/claude/signal_pattern_{tz_label}.png'
        
        # Корректируем целевой период
        tz_diff = timezone_offset - 7
        adjusted_start = target_start - timedelta(hours=tz_diff)
        adjusted_end = target_end - timedelta(hours=tz_diff)
        
        # Собираем данные
        timestamps = []
        values = []
        
        for record in self.records_2704:
            ts = datetime.fromtimestamp(record['action_time'] / 1000.0)
            ts_adjusted = ts + timedelta(hours=timezone_offset)
            
            if adjusted_start <= ts_adjusted <= adjusted_end:
                timestamps.append(ts_adjusted)
                values.append(self._extract_uint16(record['data']))
        
        if not timestamps:
            print(f"⚠ Нет данных для таймзоны UTC{timezone_offset:+d}")
            return
        
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # График 1: Значения сигнала
        ax1.plot(timestamps, values, '-o', linewidth=1.5, markersize=3, 
                alpha=0.7, color='steelblue', label='Значение uint16')
        ax1.axhline(y=1100, color='red', linestyle='--', linewidth=2, 
                   label='Threshold = 1100', alpha=0.8)
        ax1.fill_between(timestamps, 0, values, alpha=0.2, color='steelblue')
        
        # Зоны
        ax1.axhspan(0, 1100, alpha=0.15, color='green')
        ax1.axhspan(1100, max(values + [2500]), alpha=0.15, color='orange')
        
        # Аннотации зон
        ax1.text(timestamps[0], 500, 'EMPTY зона', fontsize=10, 
                color='darkgreen', fontweight='bold', alpha=0.7)
        ax1.text(timestamps[0], 1600, 'FULL зона', fontsize=10, 
                color='darkorange', fontweight='bold', alpha=0.7)
        
        ax1.set_ylabel('Значение uint16', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Паттерн сигнала frame 2704 для таймзоны UTC{timezone_offset:+d}\n'
            f'Период: {adjusted_start.strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{adjusted_end.strftime("%Y-%m-%d %H:%M:%S")} '
            f'({len(timestamps)} записей)',
            fontsize=14, fontweight='bold', pad=20
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.set_ylim([0, max(values + [2500])])
        
        # График 2: Состояние (выше/ниже порога)
        states = [1 if v > 1100 else 0 for v in values]
        ax2.fill_between(timestamps, 0, states, step='post', alpha=0.6, color='orange', 
                        label='Выше порога (FULL)')
        ax2.fill_between(timestamps, states, [1]*len(states), step='post', alpha=0.6, 
                        color='green', label='Ниже порога (EMPTY)')
        
        ax2.set_ylabel('Состояние', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Время', fontsize=12, fontweight='bold')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['EMPTY', 'FULL'])
        ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax2.legend(fontsize=10, loc='upper right')
        
        # Форматирование оси X для обоих графиков
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Статистика
        stats_text = (
            f'Статистика:\n'
            f'Мин: {min(values)}\n'
            f'Макс: {max(values)}\n'
            f'Среднее: {np.mean(values):.1f}\n'
            f'Медиана: {np.median(values):.1f}\n'
            f'Выше порога: {sum(states)}/{len(states)} ({100*sum(states)/len(states):.1f}%)'
        )
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Детальный график сохранен: {output_file}")
        plt.close()
    
    def create_comparison_plot(
        self,
        timezones_to_compare: List[int],
        target_start: datetime,
        target_end: datetime,
        output_file: str = '/home/claude/signal_comparison.png'
    ):
        """
        Создает сравнительный график для нескольких таймзон на одном графике.
        """
        fig, ax = plt.subplots(figsize=(18, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(timezones_to_compare)))
        
        for idx, tz_offset in enumerate(timezones_to_compare):
            # Корректируем целевой период
            tz_diff = tz_offset - 7
            adjusted_start = target_start - timedelta(hours=tz_diff)
            adjusted_end = target_end - timedelta(hours=tz_diff)
            
            # Собираем данные
            timestamps = []
            values = []
            
            for record in self.records_2704:
                ts = datetime.fromtimestamp(record['action_time'] / 1000.0)
                ts_adjusted = ts + timedelta(hours=tz_offset)
                
                if adjusted_start <= ts_adjusted <= adjusted_end:
                    timestamps.append(ts_adjusted)
                    values.append(self._extract_uint16(record['data']))
            
            if timestamps:
                ax.plot(timestamps, values, '-', linewidth=2, alpha=0.7, 
                       color=colors[idx], 
                       label=f'UTC{tz_offset:+d} ({len(timestamps)} записей)')
        
        ax.axhline(y=1100, color='red', linestyle='--', linewidth=2.5, 
                  label='Threshold = 1100', alpha=0.8, zorder=10)
        
        ax.set_xlabel('Время', fontsize=12, fontweight='bold')
        ax.set_ylabel('Значение uint16', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Сравнение паттернов сигнала frame 2704 по таймзонам\n'
            f'Целевой период (UTC+7): {target_start.strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{target_end.strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Сравнительный график сохранен: {output_file}")
        plt.close()


def main():
    """Основная функция"""
    
    print("=== Визуализация паттернов сигнала CAN frame 2704 ===\n")
    
    # Настройки
    log_file = "/home/claude/sample_can_log.txt"
    
    # Проверка существования файла
    import os
    if not os.path.exists(log_file):
        # Проверяем альтернативный путь
        alt_log_file = "/mnt/user-data/uploads/can_log.txt"
        if os.path.exists(alt_log_file):
            log_file = alt_log_file
        else:
            print(f"✗ Файл {log_file} не найден")
            print("Сначала запустите can_log_demo.py или загрузите свой файл")
            return
    
    # Целевой период (в UTC+7)
    target_start = datetime(2026, 1, 26, 7, 0, 0)
    target_end = datetime(2026, 1, 26, 7, 8, 0)
    
    print(f"Файл логов: {log_file}")
    print(f"Целевой период (UTC+7):")
    print(f"  От: {target_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  До: {target_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Создаем визуализатор
    visualizer = SignalPatternVisualizer(log_file)
    
    if len(visualizer.records_2704) == 0:
        print("✗ Нет записей с frame_name='2704' в файле")
        return
    
    # 1. Сетка графиков для всех таймзон
    print("\n1. Создание сетки графиков для всех таймзон...")
    visualizer.create_signal_plots(
        target_start=target_start,
        target_end=target_end,
        tz_range=(-7, 7)
    )
    
    # 2. Детальные графики для таймзон с данными
    print("\n2. Создание детальных графиков...")
    
    # Определяем таймзоны с данными
    timezones_with_data = []
    for tz_offset in range(-7, 8):
        tz_diff = tz_offset - 7
        adjusted_start = target_start - timedelta(hours=tz_diff)
        adjusted_end = target_end - timedelta(hours=tz_diff)
        
        count = 0
        for record in visualizer.records_2704:
            ts = datetime.fromtimestamp(record['action_time'] / 1000.0)
            ts_adjusted = ts + timedelta(hours=tz_offset)
            if adjusted_start <= ts_adjusted <= adjusted_end:
                count += 1
        
        if count > 0:
            timezones_with_data.append(tz_offset)
            print(f"  - UTC{tz_offset:+d}: {count} записей")
    
    # Создаем детальные графики для таймзон с данными
    for tz in timezones_with_data[:3]:  # Ограничиваем первыми 3-мя
        visualizer.create_detailed_signal_plot(
            timezone_offset=tz,
            target_start=target_start,
            target_end=target_end
        )
    
    # 3. Сравнительный график (если есть несколько таймзон с данными)
    if len(timezones_with_data) > 1:
        print("\n3. Создание сравнительного графика...")
        visualizer.create_comparison_plot(
            timezones_to_compare=timezones_with_data[:5],  # Максимум 5
            target_start=target_start,
            target_end=target_end
        )
    
    print("\n✓ Визуализация паттернов завершена!")
    print("\nСозданные файлы:")
    print("  • signal_patterns_all.png - сетка графиков для всех таймзон")
    if timezones_with_data:
        for tz in timezones_with_data[:3]:
            tz_label = f"UTC{tz:+d}".replace('+', 'plus').replace('-', 'minus')
            print(f"  • signal_pattern_{tz_label}.png - детальный график")
    if len(timezones_with_data) > 1:
        print("  • signal_comparison.png - сравнительный график")


if __name__ == "__main__":
    main()
