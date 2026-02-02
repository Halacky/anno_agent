#!/usr/bin/env python3
"""
Демонстрационный скрипт с примером данных.
Используйте это для тестирования, если у вас еще нет файла логов.
"""

from datetime import datetime, timedelta
import random


def generate_sample_log(output_file: str = '/home/claude/sample_can_log.txt'):
    """
    Генерирует пример CAN-лога для тестирования.
    Симулирует события переключения состояния.
    """
    
    print("Генерация примера CAN-лога...")
    
    # Базовое время - 26 января 2026, 00:00:00 UTC
    base_time = datetime(2026, 1, 26, 0, 0, 0)
    base_timestamp = int(base_time.timestamp() * 1000)
    
    records = []
    
    # Симулируем работу в течение 12 часов
    current_time = base_timestamp
    state = "EMPTY"  # начальное состояние
    
    for i in range(1000):
        # Каждые 500-520 мс новая запись
        current_time += random.randint(500, 520)
        
        # Определяем значение uint16 на основе состояния
        if state == "EMPTY":
            # В состоянии EMPTY значение обычно < 1100
            value = random.randint(200, 1000)
            
            # С вероятностью 5% начинаем переход в FULL
            if random.random() < 0.05:
                state = "TRANSITION_TO_FULL"
                transition_counter = 0
        
        elif state == "TRANSITION_TO_FULL":
            # Значение > 1100 для перехода
            value = random.randint(1150, 2000)
            transition_counter += 1
            
            # После 15+ последовательных значений > 1100 переходим в FULL
            if transition_counter >= 15:
                state = "FULL"
        
        elif state == "FULL":
            # В состоянии FULL значение обычно > 1100
            value = random.randint(1200, 2500)
            
            # С вероятностью 5% начинаем переход в EMPTY
            if random.random() < 0.05:
                state = "TRANSITION_TO_EMPTY"
                transition_counter = 0
        
        elif state == "TRANSITION_TO_EMPTY":
            # Значение < 1100 для перехода
            value = random.randint(200, 1000)
            transition_counter += 1
            
            # После 15+ последовательных значений < 1100 переходим в EMPTY
            if transition_counter >= 15:
                state = "EMPTY"
        
        # Формируем байты данных
        byte5 = value & 0xFF
        byte6 = (value >> 8) & 0xFF
        
        # Генерируем случайные байты для остальных позиций
        data_bytes = [
            0xAA, 0xE8, 0x27, 0x04, 0xFF, 
            byte5, byte6,
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            0xFF, 0x55
        ]
        
        data_str = ' '.join(f'{b:02X}' for b in data_bytes)
        
        # Создаем запись
        record = f"{{'action_time': {current_time}, 'data': '{data_str}', 'frame_name': '2704'}}"
        records.append(record)
        
        # Иногда добавляем записи 2717
        if random.random() < 0.3:
            current_time += random.randint(10, 50)
            data_2717 = "AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55"
            record_2717 = f"{{'action_time': {current_time}, 'data': '{data_2717}', 'frame_name': '2717'}}"
            records.append(record_2717)
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(record + '\n')
    
    print(f"✓ Создано {len(records)} записей")
    print(f"✓ Сохранено в {output_file}")
    
    # Показываем примеры записей
    print("\nПримеры записей:")
    for i in [0, len(records)//2, -1]:
        print(f"  {records[i]}")


def print_usage_instructions():
    """Выводит инструкции по использованию"""
    
    print("\n" + "=" * 80)
    print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ")
    print("=" * 80 + "\n")
    
    print("1. Если у вас есть файл с CAN-логами:")
    print("   • Загрузите его в чат")
    print("   • Откройте can_log_analyzer.py")
    print("   • Измените переменную log_file на путь к вашему файлу")
    print("   • Запустите: python3 can_log_analyzer.py\n")
    
    print("2. Если вы хотите протестировать на примере данных:")
    print("   • Запустите: python3 can_log_demo.py")
    print("   • Затем запустите: python3 can_log_analyzer.py")
    print("   • (analyzer автоматически использует sample_can_log.txt если основной файл не найден)\n")
    
    print("3. Для визуализации результатов:")
    print("   • После запуска analyzer запустите: python3 can_log_visualizer.py")
    print("   • Будут созданы графики и детальный отчет\n")
    
    print("4. Настройка целевого периода:")
    print("   • Откройте can_log_analyzer.py")
    print("   • Найдите строки с target_start и target_end")
    print("   • Измените даты на нужные вам\n")
    
    print("ФАЙЛЫ РЕЗУЛЬТАТОВ:")
    print("  • timezone_analysis_results.json - JSON с результатами")
    print("  • timezone_summary.png - график сводки")
    print("  • timezone_timeline.png - график временных линий")
    print("  • timezone_report.txt - детальный текстовый отчет")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    generate_sample_log()
    print_usage_instructions()
