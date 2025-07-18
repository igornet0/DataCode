# 🎪 Демонстрации DataCode

Этот раздел содержит комплексные демонстрации, показывающие все возможности языка DataCode в действии.

## 📋 Содержание

### 1. `showcase.dc` - Полная демонстрация языка
**Описание**: Комплексный пример, демонстрирующий все основные возможности DataCode в одном файле.

**Что демонстрируется**:
- Все типы данных
- Пользовательские функции
- Условные конструкции
- Циклы и итерации
- Работа с массивами
- Табличные операции
- Рекурсивные алгоритмы
- Обработка ошибок
- Встроенные функции
- Файловые операции

**Цель**: Показать полную мощь и возможности языка DataCode в реальном сценарии использования.

## 🎯 Как запускать

```bash
# Полная демонстрация (может занять время)
cargo run examples/07-демонстрации/showcase.dc

# Если DataCode установлен глобально
datacode examples/07-демонстрации/showcase.dc
```

## 📚 Что включает showcase.dc

### Базовые возможности
- Объявление переменных всех типов
- Арифметические и логические операции
- Работа со строками

### Функции и алгоритмы
- Пользовательские функции с параметрами
- Рекурсивные алгоритмы (факториал, Фибоначчи)
- Функции высшего порядка

### Структуры данных
- Массивы (простые и вложенные)
- Смешанные типы данных
- Индексирование и манипуляции

### Условная логика
- Простые и сложные условия
- Вложенные условия
- Логические операторы

### Циклы и итерации
- Циклы for...in
- Итерация по массивам
- Перечисление с индексами

### Работа с данными
- Загрузка CSV файлов
- Табличные операции
- Фильтрация и сортировка
- Агрегация данных

### Встроенные функции
- Математические функции
- Строковые операции
- Функции массивов
- Файловые операции

## 🔍 Структура демонстрации

### Раздел 1: Основы
```datacode
# Переменные и типы данных
global name = 'DataCode'
global version = 1.0
global active = true
```

### Раздел 2: Функции
```datacode
# Пользовательские функции
global function greet(name) do
    return 'Hello, ' + name + '!'
endfunction
```

### Раздел 3: Алгоритмы
```datacode
# Рекурсивные алгоритмы
global function fibonacci(n) do
    if n <= 1 do
        return n
    endif
    return fibonacci(n-1) + fibonacci(n-2)
endfunction
```

### Раздел 4: Структуры данных
```datacode
# Массивы и таблицы
global data = [1, 2, 3, 'mixed', true]
global matrix = [[1, 2], [3, 4]]
```

### Раздел 5: Обработка данных
```datacode
# Работа с CSV и таблицами
global employees = read_file('data.csv')
global filtered = table_select(employees, ['name', 'salary'])
```

## ⚡ Производительность

**Предупреждение**: `showcase.dc` выполняет множество операций и может занять значительное время, особенно:
- Рекурсивные вычисления
- Операции с большими массивами
- Загрузка и обработка файлов
- Сложные табличные операции

## 💡 Как использовать showcase.dc

### Для изучения
1. **Читайте код по разделам** - не пытайтесь понять все сразу
2. **Запускайте по частям** - комментируйте сложные разделы
3. **Экспериментируйте** - изменяйте параметры и значения
4. **Изучайте вывод** - анализируйте результаты выполнения

### Для демонстрации
1. **Покажите возможности** - используйте как презентацию языка
2. **Объясните концепции** - комментируйте ключевые моменты
3. **Сравните с другими языками** - покажите уникальные особенности
4. **Подчеркните простоту** - обратите внимание на читаемость кода

### Для тестирования
1. **Проверьте установку** - убедитесь, что все работает
2. **Тестируйте производительность** - измерьте время выполнения
3. **Найдите узкие места** - определите медленные операции
4. **Оптимизируйте код** - улучшите производительность

## 🔗 Навигация

### Все предыдущие разделы
Showcase.dc объединяет концепции из всех разделов - убедитесь, что изучили их перед запуском:
- **[01-основы](../01-основы/)** - базовые операции и переменные
- **[02-синтаксис-языка](../02-синтаксис-языка/)** - функции, условия и циклы
- **[05-типы-данных](../05-типы-данных/)** - система типов и isinstance()
- **[04-обработка-данных](../04-обработка-данных/)** - табличные операции и CSV
- **[03-продвинутые-возможности](../03-продвинутые-возможности/)** - рекурсия и обработка ошибок
- **[06-инструменты-разработки](../06-инструменты-разработки/)** - отладка и тестирование

### Дополнительные ресурсы
- **[../INDEX.md](../INDEX.md)** - 📋 Быстрый индекс всех примеров
- **[../README.md](../README.md)** - 📚 Главная страница примеров
- **[../../README.md](../../README.md)** - 🏠 Основная документация DataCode

## 📈 Рекомендации по изучению

### Перед запуском showcase.dc изучите:
1. **Основы** - переменные и функции
2. **Синтаксис** - условия и циклы
3. **Типы данных** - система типов
4. **Обработку данных** - табличные операции

### После изучения showcase.dc:
1. **Создайте свой проект** - используйте изученные концепции
2. **Оптимизируйте код** - улучшите производительность
3. **Добавьте функциональность** - расширьте возможности
4. **Поделитесь опытом** - помогите другим изучить DataCode

## ⚠️ Важные замечания

- **Время выполнения**: Может быть значительным из-за сложности
- **Ресурсы**: Потребляет память и процессорное время
- **Зависимости**: Может требовать наличия файлов данных
- **Отладка**: Используйте по частям для понимания проблем

## 🎓 Образовательная ценность

Showcase.dc служит как:
- **Учебное пособие** - демонстрирует все возможности
- **Справочник** - показывает правильное использование
- **Тест производительности** - проверяет возможности системы
- **Источник вдохновения** - дает идеи для собственных проектов

---

**Showcase.dc - это кульминация изучения DataCode!** 🎪✨
