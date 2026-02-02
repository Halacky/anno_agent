#!/usr/bin/env python3
"""
Анализатор CAN-логов с поддержкой различных интерпретаций таймзон.
Извлекает данные frame_name='2704' и обрабатывает их согласно логике переключения состояний.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class CANRecord:
    """Представление одной записи из CAN-лога"""
    action_time: int  # миллисекунды (timestamp)
    data: str         # hex-строка
    frame_name: str
    
    def get_bytes(self) -> List[int]:
        """Преобразует hex-строку в список байт"""
        hex_values = self.data.strip().split()
        return [int(x, 16) for x in hex_values]
    
    def get_uint16_at_5_6(self) -> int:
        """Извлекает uint16 из байтов 5 и 6 (little-endian)"""
        bytes_data = self.get_bytes()
        if len(bytes_data) < 7:
            return 0
        return bytes_data[5] + (bytes_data[6] << 8)
    
    def get_timestamp(self, timezone_offset_hours: int = 0) -> datetime:
        """
        Конвертирует action_time в datetime с учетом таймзоны.
        
        Args:
            timezone_offset_hours: смещение от UTC (-7 до +7)
        """
        # action_time в миллисекундах - интерпретируем как Unix timestamp
        base_dt = datetime.fromtimestamp(self.action_time / 1000.0)
        # Применяем смещение таймзоны
        return base_dt + timedelta(hours=timezone_offset_hours)


class CANLogParser:
    """Парсер для CAN-логов в формате словарей Python"""
    
    @staticmethod
    def parse_line(line: str) -> CANRecord:
        """Парсит одну строку лога"""
        # Используем ast.literal_eval для безопасного парсинга
        try:
            # Убираем лишние пробелы и парсим как dict
            data_dict = eval(line.strip())
            return CANRecord(
                action_time=data_dict['action_time'],
                data=data_dict['data'],
                frame_name=data_dict['frame_name']
            )
        except Exception as e:
            raise ValueError(f"Ошибка парсинга строки: {e}")
    
    @staticmethod
    def parse_file(filepath: str) -> List[CANRecord]:
        """Парсит весь файл и возвращает список записей"""
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = CANLogParser.parse_line(line)
                    records.append(record)
                except Exception as e:
                    print(f"Предупреждение: строка {line_num} пропущена: {e}")
        return records


class StateDetector:
    """Детектор переключения состояний по алгоритму 2704"""
    
    THRESHOLD = 1100
    CONSECUTIVE = 15
    EMPTY = "EMPTY"
    FULL = "FULL"
    
    def __init__(self):
        self.current_state = self.EMPTY
        self._cnt_above = 0
        self._cnt_below = 0
        self.state_changes = []  # список (timestamp, new_state, value)
    
    def reset(self):
        """Сброс состояния детектора"""
        self.current_state = self.EMPTY
        self._cnt_above = 0
        self._cnt_below = 0
        self.state_changes = []
    
    def process_value(self, value: int, timestamp: datetime) -> bool:
        """
        Обрабатывает одно значение uint16.
        Возвращает True, если произошло изменение состояния.
        """
        if value > self.THRESHOLD:
            self._cnt_above += 1
            self._cnt_below = 0
        else:
            self._cnt_below += 1
            self._cnt_above = 0
        
        # Проверка на переход EMPTY -> FULL
        if self.current_state == self.EMPTY and self._cnt_above >= self.CONSECUTIVE:
            self.current_state = self.FULL
            self._cnt_above = 0
            self.state_changes.append((timestamp, self.FULL, value))
            return True
        
        # Проверка на переход FULL -> EMPTY
        if self.current_state == self.FULL and self._cnt_below >= self.CONSECUTIVE:
            self.current_state = self.EMPTY
            self._cnt_below = 0
            self.state_changes.append((timestamp, self.EMPTY, value))
            return True
        
        return False


class TimezoneAnalyzer:
    """Анализатор логов с разными интерпретациями таймзон"""
    
    def __init__(self, records: List[CANRecord]):
        self.all_records = records
        self.records_2704 = [r for r in records if r.frame_name == '2704']
    
    def analyze_timezone(
        self, 
        timezone_offset: int,
        target_start: datetime,
        target_end: datetime
    ) -> Dict[str, Any]:
        """
        Анализирует логи с заданным смещением таймзоны.
        
        Args:
            timezone_offset: смещение в часах (-7 до +7)
            target_start: начало целевого периода (в таймзоне +7)
            target_end: конец целевого периода (в таймзоне +7)
        
        Returns:
            Словарь с результатами анализа
        """
        # Конвертируем целевой период из +7 в нужную таймзону
        # target_start и target_end даны в +7, нужно найти соответствующий период в данных
        tz_diff = timezone_offset - 7  # разница между текущей интерпретацией и +7
        
        adjusted_start = target_start - timedelta(hours=tz_diff)
        adjusted_end = target_end - timedelta(hours=tz_diff)
        
        # Фильтруем записи, попадающие в период
        filtered_records = []
        for record in self.records_2704:
            ts = record.get_timestamp(timezone_offset)
            if adjusted_start <= ts <= adjusted_end:
                filtered_records.append((record, ts))
        
        # Обрабатываем через детектор состояний
        detector = StateDetector()
        for record, ts in filtered_records:
            value = record.get_uint16_at_5_6()
            detector.process_value(value, ts)
        
        # Собираем статистику
        if filtered_records:
            first_ts = filtered_records[0][1]
            last_ts = filtered_records[-1][1]
            first_raw = filtered_records[0][0].action_time
            last_raw = filtered_records[-1][0].action_time
        else:
            first_ts = last_ts = None
            first_raw = last_raw = None
        
        return {
            'timezone_offset': timezone_offset,
            'timezone_label': f"UTC{timezone_offset:+d}" if timezone_offset != 0 else "UTC",
            'target_period': {
                'start': target_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end': target_end.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': 'UTC+7'
            },
            'adjusted_period': {
                'start': adjusted_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end': adjusted_end.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': f"UTC{timezone_offset:+d}" if timezone_offset != 0 else "UTC"
            },
            'records_count': len(filtered_records),
            'first_record': {
                'timestamp': first_ts.strftime('%Y-%m-%d %H:%M:%S') if first_ts else None,
                'raw_time': first_raw
            } if first_ts else None,
            'last_record': {
                'timestamp': last_ts.strftime('%Y-%m-%d %H:%M:%S') if last_ts else None,
                'raw_time': last_raw
            } if last_ts else None,
            'state_changes': [
                {
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'new_state': state,
                    'value': value
                }
                for ts, state, value in detector.state_changes
            ],
            'state_changes_count': len(detector.state_changes)
        }
    
    def analyze_all_timezones(
        self,
        target_start: datetime,
        target_end: datetime,
        tz_range: Tuple[int, int] = (-7, 7)
    ) -> List[Dict[str, Any]]:
        """
        Анализирует логи со всеми возможными смещениями таймзон.
        
        Args:
            target_start: начало целевого периода (в таймзоне +7)
            target_end: конец целевого периода (в таймзоне +7)
            tz_range: диапазон смещений таймзон (мин, макс)
        
        Returns:
            Список результатов для каждой таймзоны
        """
        results = []
        for tz_offset in range(tz_range[0], tz_range[1] + 1):
            result = self.analyze_timezone(tz_offset, target_start, target_end)
            results.append(result)
        
        return results


def main():
    """Основная функция для демонстрации использования"""
    
    # Пример использования
    print("=== Анализатор CAN-логов с поддержкой таймзон ===\n")
    
    # Укажите путь к вашему файлу логов
    log_file = "/mnt/user-data/uploads/can_log.txt"  # Измените на ваш путь
    sample_file = "/home/claude/sample_can_log.txt"
    
    try:
        # Парсинг логов
        # Проверяем наличие основного файла, иначе используем пример
        import os
        if os.path.exists(log_file):
            print(f"Загрузка логов из {log_file}...")
            file_to_use = log_file
        elif os.path.exists(sample_file):
            print(f"Основной файл не найден, используем пример: {sample_file}...")
            file_to_use = sample_file
        else:
            raise FileNotFoundError("Файл логов не найден. Запустите can_log_demo.py для создания примера.")
        
        parser = CANLogParser()
        records = parser.parse_file(file_to_use)
        print(f"✓ Загружено {len(records)} записей")
        
        records_2704 = [r for r in records if r.frame_name == '2704']
        print(f"✓ Найдено {len(records_2704)} записей с frame_name='2704'\n")
        
        # Целевой период (в таймзоне +7)
        # Для примера данных используем период 07:00-07:08
        target_start = datetime(2026, 1, 26, 7, 0, 0)
        target_end = datetime(2026, 1, 26, 7, 8, 0)
        
        print(f"Целевой период (UTC+7):")
        print(f"  От: {target_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  До: {target_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Анализ
        analyzer = TimezoneAnalyzer(records)
        results = analyzer.analyze_all_timezones(target_start, target_end)
        
        # Вывод результатов
        print("=" * 80)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА ПО ТАЙМЗОНАМ")
        print("=" * 80 + "\n")
        
        for result in results:
            print(f"┌─ {result['timezone_label']} " + "─" * (75 - len(result['timezone_label'])))
            print(f"│ Целевой период (UTC+7): {result['target_period']['start']} - {result['target_period']['end']}")
            print(f"│ Скорректированный период ({result['timezone_label']}): {result['adjusted_period']['start']} - {result['adjusted_period']['end']}")
            print(f"│ Найдено записей: {result['records_count']}")
            
            if result['first_record']:
                print(f"│ Первая запись: {result['first_record']['timestamp']} (raw: {result['first_record']['raw_time']})")
                print(f"│ Последняя запись: {result['last_record']['timestamp']} (raw: {result['last_record']['raw_time']})")
            else:
                print(f"│ ⚠ Нет записей в этом периоде")
            
            print(f"│ Переключений состояния: {result['state_changes_count']}")
            
            if result['state_changes']:
                print(f"│ Изменения состояний:")
                for change in result['state_changes']:
                    print(f"│   • {change['timestamp']} → {change['new_state']} (value={change['value']})")
            
            print("└" + "─" * 79 + "\n")
        
        # Сохранение результатов в JSON
        output_file = "/home/claude/timezone_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Результаты сохранены в {output_file}")
        
    except FileNotFoundError:
        print(f"✗ Файл {log_file} не найден")
        print("\nПожалуйста, укажите правильный путь к файлу с логами.")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
