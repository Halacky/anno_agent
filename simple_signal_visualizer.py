#!/usr/bin/env python3
"""
Простая визуализация сигнала frame 2704 для каждой таймзоны.
Показывает только исходные значения uint16 (байты 5-6) без дополнительной логики.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np


class SimpleSignalVisualizer:
    """Простой визуализатор сигнала по таймзонам"""
    
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
        """Извлекает uint16 из hex-строки данных (байты 5 и 6)"""
        hex_values = data_str.strip().split()
        if len(hex_values) < 7:
            return 0
        byte5 = int(hex_values[5], 16)
        byte6 = int(hex_values[6], 16)
        return byte5 + (byte6 << 8)
    
    def create_all_timezones_grid(
        self,
        target_start: datetime,
        target_end: datetime,
        tz_range: tuple = (-7, 7),
        output_file: str = '/home/claude/signal_all_timezones.png'
    ):
        """
        Создает сетку графиков с сигналом для всех таймзон.
        """
        timezones = list(range(tz_range[0], tz_range[1] + 1))
        n_timezones = len(timezones)
        
        # Создаем сетку графиков (5 колонок)
        n_cols = 5
        n_rows = (n_timezones + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = axes.flatten() if n_timezones > 1 else [axes]
        
        for idx, tz_offset in enumerate(timezones):
            ax = axes[idx]
            
            # Корректируем целевой период для текущей таймзоны
            tz_diff = tz_offset - 7  # разница с UTC+7
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
                ax.plot(timestamps, values, '-', linewidth=0.8, color='steelblue')
                ax.fill_between(timestamps, 0, values, alpha=0.3, color='steelblue')
                
                tz_label = f'UTC{tz_offset:+d}' if tz_offset != 0 else 'UTC'
                ax.set_title(f'{tz_label}\n{len(timestamps)} записей', 
                           fontweight='bold', fontsize=10)
                ax.set_ylabel('uint16', fontsize=8)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='both', labelsize=7)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Показываем диапазон значений
                ax.text(0.02, 0.98, f'min: {min(values)}\nmax: {max(values)}', 
                       transform=ax.transAxes, fontsize=7, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'Нет данных', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                tz_label = f'UTC{tz_offset:+d}' if tz_offset != 0 else 'UTC'
                ax.set_title(f'{tz_label}\n0 записей', 
                           fontweight='bold', fontsize=10, color='gray')
                ax.set_xlim([adjusted_start, adjusted_end])
                ax.set_ylim([0, 2500])
        
        # Скрываем лишние подграфики
        for idx in range(n_timezones, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(
            f'Сигнал frame 2704 (uint16 из байтов 5-6) по таймзонам\n'
            f'Целевой период (UTC+7): {target_start.strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{target_end.strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {output_file}")
        plt.close()
    
    def create_detailed_plot(
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
            output_file = f'/home/claude/signal_detailed_{tz_label}.png'
        
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
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # График сигнала
        ax.plot(timestamps, values, '-o', linewidth=1.5, markersize=2, 
                color='steelblue', label='Сигнал uint16')
        ax.fill_between(timestamps, 0, values, alpha=0.2, color='steelblue')
        
        tz_label = f'UTC{timezone_offset:+d}' if timezone_offset != 0 else 'UTC'
        ax.set_ylabel('Значение uint16 (байты 5-6)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Время', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Сигнал frame 2704 для таймзоны {tz_label}\n'
            f'Период: {adjusted_start.strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{adjusted_end.strftime("%Y-%m-%d %H:%M:%S")} '
            f'({len(timestamps)} записей)',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
        
        # Форматирование оси X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Статистика
        stats_text = (
            f'Статистика:\n'
            f'Записей: {len(values)}\n'
            f'Мин: {min(values)}\n'
            f'Макс: {max(values)}\n'
            f'Среднее: {np.mean(values):.1f}\n'
            f'Медиана: {np.median(values):.1f}\n'
            f'Ст. откл: {np.std(values):.1f}'
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Детальный график сохранен: {output_file}")
        plt.close()
    
    def create_comparison_plot(
        self,
        timezones_to_compare: list,
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
                tz_label = f'UTC{tz_offset:+d}' if tz_offset != 0 else 'UTC'
                ax.plot(timestamps, values, '-', linewidth=2, alpha=0.7, 
                       color=colors[idx], 
                       label=f'{tz_label} ({len(timestamps)} записей)')
        
        ax.set_xlabel('Время', fontsize=12, fontweight='bold')
        ax.set_ylabel('Значение uint16 (байты 5-6)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Сравнение сигнала frame 2704 по таймзонам\n'
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
    
    print("=== Визуализация сигнала CAN frame 2704 ===\n")
    
    # Настройки
    log_file = "/home/claude/sample_can_log.txt"
    
    # Проверка существования файла
    import os
    if not os.path.exists(log_file):
        alt_log_file = "/mnt/user-data/uploads/can_log.txt"
        if os.path.exists(alt_log_file):
            log_file = alt_log_file
        else:
            print(f"✗ Файл {log_file} не найден")
            print("Загрузите ваш файл логов или запустите can_log_demo.py для создания примера")
            return
    
    # Целевой период (в UTC+7)
    # ИЗМЕНИТЕ ЭТИ ДАТЫ НА ВАШИ:
    target_start = datetime(2026, 1, 26, 7, 0, 0)
    target_end = datetime(2026, 1, 26, 7, 8, 0)
    
    print(f"Файл логов: {log_file}")
    print(f"Целевой период (UTC+7):")
    print(f"  От: {target_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  До: {target_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Создаем визуализатор
    visualizer = SimpleSignalVisualizer(log_file)
    
    if len(visualizer.records_2704) == 0:
        print("✗ Нет записей с frame_name='2704' в файле")
        return
    
    # 1. Сетка графиков для всех таймзон
    print("1. Создание сетки графиков для всех таймзон...")
    visualizer.create_all_timezones_grid(
        target_start=target_start,
        target_end=target_end,
        tz_range=(-7, 7)
    )
    
    # 2. Определяем таймзоны с данными и создаем детальные графики
    print("\n2. Анализ таймзон и создание детальных графиков...")
    
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
            tz_label = f'UTC{tz_offset:+d}' if tz_offset != 0 else 'UTC'
            print(f"  - {tz_label}: {count} записей")
    
    # Создаем детальные графики для таймзон с данными (максимум 3)
    for tz in timezones_with_data[:3]:
        visualizer.create_detailed_plot(
            timezone_offset=tz,
            target_start=target_start,
            target_end=target_end
        )
    
    # 3. Сравнительный график (если несколько таймзон с данными)
    if len(timezones_with_data) > 1:
        print("\n3. Создание сравнительного графика...")
        visualizer.create_comparison_plot(
            timezones_to_compare=timezones_with_data[:5],
            target_start=target_start,
            target_end=target_end
        )
    
    print("\n✓ Визуализация завершена!")
    print("\nСозданные файлы:")
    print("  • signal_all_timezones.png - сетка со всеми таймзонами")
    if timezones_with_data:
        for tz in timezones_with_data[:3]:
            tz_label = f"UTC{tz:+d}".replace('+', 'plus').replace('-', 'minus')
            print(f"  • signal_detailed_{tz_label}.png - детальный график")
    if len(timezones_with_data) > 1:
        print("  • signal_comparison.png - сравнение таймзон")


if __name__ == "__main__":
    main()
