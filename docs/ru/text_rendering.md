# Отрисовка текста в модуле plot

## Обзор

Модуль `plot` использует библиотеку `fontdue` для отрисовки текста на графиках и изображениях. Система поддерживает UTF-8, включая кириллицу, и использует кэширование глифов для оптимизации производительности.

## Архитектура системы отрисовки текста

### Основные компоненты

1. **Renderer** (`src/lib/plot/renderer.rs`) - основной класс рендеринга
2. **FontAtlas** (`src/lib/plot/font_atlas.rs`) - кэш для растрированных глифов
3. **Font** (из `fontdue`) - загруженный шрифт
4. **Buffer** - буфер пикселей для отрисовки (формат BGRA)

### Поток данных

```
draw_figure() 
  → draw_text_improved() 
    → draw_text_with_font() 
      → FontAtlas.get_or_rasterize() 
        → font.rasterize() (если не в кэше)
```

## Детальное описание процесса отрисовки

### 1. Инициализация шрифта

**Местоположение:** `renderer.rs:525-692` (функция `load_font()`)

При создании `Renderer` происходит загрузка шрифта:

1. **Приоритет моноширинных шрифтов:**
   - macOS: Monaco, Menlo, Courier New
   - Linux: DejaVu Sans Mono, Liberation Mono
   - Windows: Consolas, Courier New

2. **Fallback на пропорциональные шрифты:**
   - Встроенный Inter (18pt Regular)
   - Inter Variable Font
   - Roboto (Regular и Variable)
   - Системные шрифты (SF Pro, Helvetica, Arial и т.д.)

3. **Если шрифт не найден:** используется bitmap fallback

**Настройки шрифта:**
```rust
let font_settings = fontdue::FontSettings {
    scale: 1.0,
    ..Default::default()
};
```

### 2. Определение позиции текста (baseline)

**Местоположение:** `renderer.rs:416-449` (функция `draw_figure()`)

#### Вычисление baseline

Baseline (базовая линия) - это воображаемая линия, на которой "стоят" символы. Правильное вычисление baseline критично для корректного выравнивания текста.

```rust
let text_y = if let Some(font) = self.font.as_ref() {
    if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
        // baseline_y = cell_y + ascent
        // ascent - расстояние от baseline до верха символов (положительное значение)
        cell_y + line_metrics.ascent.round() as usize
    } else {
        cell_y + 14 // fallback если метрики недоступны
    }
} else {
    cell_y + 14 // fallback если шрифт не загружен
};
```

**Важно:** 
- `cell_y` - это Y-координата верхней границы ячейки
- `line_metrics.ascent` - расстояние от baseline до верха символов
- Baseline НЕ равен верхней границе текста!

#### Метрики строки (LineMetrics)

Структура `LineMetrics` содержит:
- `ascent: f32` - расстояние от baseline до верха символов (положительное)
- `descent: f32` - расстояние от baseline до низа символов (отрицательное)
- `line_gap: f32` - дополнительный промежуток между строками
- `new_line_size: f32` - общая высота строки (ascent - descent + line_gap)

### 3. Вычисление ширины текста

**Местоположение:** `renderer.rs:694-710` (функция `calculate_text_width()`)

Для центрирования текста необходимо знать его точную ширину:

```rust
fn calculate_text_width(font: Option<&Font>, text: &str, font_size: f32) -> f32 {
    if let Some(font) = font {
        let char_width = Self::calculate_max_char_width(font, font_size);
        text.chars().count() as f32 * char_width
    } else {
        let avg_char_width = font_size * 0.6;
        text.chars().count() as f32 * avg_char_width
    }
}
```

**Особенности:**
- Для моноширинных шрифтов используется фиксированная ширина символа
- Ширина вычисляется как максимум из тестовых символов
- Для пропорциональных шрифтов используется средняя ширина

#### Вычисление максимальной ширины символа

**Местоположение:** `renderer.rs:776-795` (функция `calculate_max_char_width()`)

```rust
fn calculate_max_char_width(font: &Font, font_size: f32) -> f32 {
    let test_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz МеткаPredict:";
    let mut max_width = 0.0f32;
    
    for ch in test_chars.chars() {
        let (metrics, _) = font.rasterize(ch, font_size);
        max_width = max_width.max(metrics.advance_width);
    }
    
    max_width
}
```

Это обеспечивает единообразное расстояние между символами.

### 4. Центрирование текста

**Местоположение:** `renderer.rs:423-424`

```rust
let text_width = Self::calculate_text_width(self.font.as_ref(), title, font_size);
let text_x = (cell_x as f32 + (cell_width_usize as f32 - text_width) / 2.0).round() as usize;
```

Текст центрируется по горизонтали в ячейке.

### 5. Основная функция отрисовки текста

**Местоположение:** `renderer.rs:801-895` (функция `draw_text_with_font()`)

#### Шаг 1: Подготовка параметров

```rust
let rasterize_size = font_size;
let char_width = Self::calculate_max_char_width(font, rasterize_size);
let baseline_y = y as f32; // y уже содержит правильную baseline
let base_x = x as f32;
let mut current_x = base_x;
```

#### Шаг 2: Нормализация xmin для выравнивания

Для обеспечения единообразного выравнивания всех символов вычисляется минимальный `xmin`:

```rust
let min_xmin = text.chars()
    .map(|ch| {
        let (metrics, _) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
            font.rasterize(c, size)
        });
        metrics.xmin as f32
    })
    .fold(0.0f32, |acc, xmin| acc.min(xmin));
```

Это гарантирует, что все символы начинаются с одинаковой визуальной позиции.

#### Шаг 3: Отрисовка каждого символа

Для каждого символа в тексте:

```rust
for ch in text.chars() {
    // Получить или растрировать глиф (с кэшированием)
    let (metrics, bitmap) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
        font.rasterize(c, size)
    });
    
    // Вычислить позицию глифа
    let normalized_xmin = metrics.xmin as f32 - min_xmin;
    let glyph_x = (current_x + normalized_xmin).round() as i32;
    
    // Вычислить Y-позицию относительно baseline
    let glyph_y = (baseline_y + metrics.ymin as f32).round() as i32;
    
    // Отрисовать bitmap глифа
    // ...
    
    // Перейти к следующему символу
    current_x += char_width;
}
```

#### Формула позиционирования глифа

**Горизонтальная позиция:**
```
glyph_x = current_x + normalized_xmin
normalized_xmin = metrics.xmin - min_xmin
```

**Вертикальная позиция:**
```
glyph_y = baseline_y + metrics.ymin
```

**Важно:** 
- `metrics.ymin` может быть отрицательным для символов с descenders (p, d, q, g, y)
- Это нормально и ожидаемо - разные символы имеют разные `ymin`
- При правильной baseline все символы выравниваются корректно

#### Шаг 4: Отрисовка bitmap глифа

```rust
if metrics.width > 0 {
    for (row_idx, row) in bitmap.chunks(metrics.width).enumerate() {
        let pixel_y = (glyph_y + row_idx as i32) as usize;
        if pixel_y >= buffer_height {
            break;
        }
        
        for (col_idx, &alpha) in row.iter().enumerate() {
            let pixel_x = glyph_x + col_idx as i32;
            if pixel_x < 0 || pixel_x as usize >= buffer_width {
                continue;
            }
            
            if alpha > 0 {
                // Альфа-блендинг с фоном
                let idx = pixel_y * buffer_width + pixel_x as usize;
                if idx < buffer.len() {
                    let current_pixel = buffer[idx];
                    buffer[idx] = Self::blend(current_pixel, 255, 255, 255, alpha);
                }
            }
        }
    }
}
```

### 6. Альфа-блендинг

**Местоположение:** `renderer.rs:713-734` (функция `blend()`)

Текст отрисовывается с альфа-каналом для плавного сглаживания:

```rust
fn blend(bg: u32, fg_r: u8, fg_g: u8, fg_b: u8, alpha: u8) -> u32 {
    let a = alpha as u32;
    let inv_a = 255 - a;
    
    // Извлечь компоненты фона (формат BGRA: 0xAABBGGRR)
    let bg_r = (bg >> 0) & 0xFF;
    let bg_g = (bg >> 8) & 0xFF;
    let bg_b = (bg >> 16) & 0xFF;
    
    // Целочисленный блендинг: (fg * a + bg * inv_a) / 255
    let r = ((fg_r as u32 * a + bg_r * inv_a) / 255) as u8;
    let g = ((fg_g as u32 * a + bg_g * inv_a) / 255) as u8;
    let b = ((fg_b as u32 * a + bg_b * inv_a) / 255) as u8;
    
    // Формат: 0xAABBGGRR
    (0xFF << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
}
```

**Формат пикселя:** BGRA (Blue, Green, Red, Alpha)
- Текст отрисовывается белым цветом (255, 255, 255)
- Альфа-канал обеспечивает сглаживание краев

### 7. Кэширование глифов (FontAtlas)

**Местоположение:** `font_atlas.rs`

`FontAtlas` кэширует растрированные глифы для оптимизации производительности:

```rust
pub struct FontAtlas {
    cache: HashMap<(char, u32), GlyphData>,
}
```

**Ключ кэша:** `(символ, округленный_размер_шрифта)`

**Округление размера:**
```rust
let rounded_size = (font_size * 2.0).round() as u32;
```

Это уменьшает размер кэша, сохраняя качество (например, 32.0, 32.5, 33.0).

**Использование:**
```rust
let (metrics, bitmap) = atlas.get_or_rasterize(ch, rasterize_size, |c, size| {
    font.rasterize(c, size)
});
```

Если глиф уже в кэше, он возвращается без повторной растризации.

### 8. Fallback: Bitmap шрифт

**Местоположение:** `renderer.rs:897-936` (функция `draw_text_bitmap()`)

Если шрифт не загружен, используется простой bitmap шрифт:

```rust
fn draw_text_bitmap(
    buffer: &mut [u32],
    x: usize,
    y: usize,
    text: &str,
    buffer_width: usize,
    buffer_height: usize,
) {
    let font_width = 16;
    let font_height = 24;
    let scale = 2;
    
    // Отрисовка каждого символа через паттерны
    for ch in text.chars().take(50) {
        Self::draw_char_scaled(buffer, current_x, y, ch, ...);
        current_x += font_width * scale + 2;
    }
}
```

Паттерны символов определены в `get_char_pattern()` (строки 976-1462).

## Координатная модель fontdue

### Baseline-ориентированная модель

В fontdue используется baseline-ориентированная координатная модель:

1. **Baseline** - это фиксированная горизонтальная линия для всей строки
2. **metrics.ymin** - минимальная Y-координата глифа относительно baseline (может быть отрицательной)
3. **metrics.ymax** - максимальная Y-координата глифа относительно baseline (положительная)
4. **metrics.xmin** - минимальная X-координата глифа относительно начала символа
5. **metrics.advance_width** - ширина символа (расстояние до следующего символа)

### Важные метрики глифа

```rust
struct Metrics {
    advance_width: f32,  // Ширина символа
    advance_height: f32, // Высота символа
    bearing_x: f32,      // Горизонтальное смещение
    bearing_y: f32,      // Вертикальное смещение
    xmin: i32,           // Минимальная X-координата bounding box
    ymin: i32,           // Минимальная Y-координата bounding box (относительно baseline)
    xmax: i32,           // Максимальная X-координата bounding box
    ymax: i32,           // Максимальная Y-координата bounding box (относительно baseline)
    width: usize,        // Ширина bitmap
    height: usize,       // Высота bitmap
}
```

## Особенности реализации

### 1. DPI-aware рендеринг

Размер шрифта масштабируется с учетом DPI экрана:

```rust
let scale_factor = window_ref.scale_factor() as f32;
let font_size = base_font_size * self.scale_factor;
```

### 2. Моноширинные vs пропорциональные шрифты

- **Моноширинные:** все символы имеют одинаковую ширину (`advance_width`)
- **Пропорциональные:** ширина символов разная

Для моноширинных шрифтов используется фиксированная ширина символа для единообразного интервала.

### 3. Нормализация xmin

Для обеспечения единообразного выравнивания все символы нормализуются по минимальному `xmin`:

```rust
let normalized_xmin = metrics.xmin as f32 - min_xmin;
```

Это гарантирует, что все символы начинаются с одинаковой визуальной позиции.

### 4. Разные ymin для разных символов

**Это нормально!** Разные символы имеют разные `ymin`:
- Символы без descenders (x, a, e, o): `ymin ≈ -2`
- Символы с descenders (p, d, q, g, y): `ymin ≈ -8`
- Цифры: `ymin ≈ -1`
- Знаки препинания: могут иметь свои значения

При правильной baseline все символы выравниваются корректно.

## Проблемы и решения

### Проблема: "Плавающий" текст

**Причина:** Неправильная установка baseline (использование `cell_y + 14` вместо метрик шрифта)

**Решение:** Использовать `font.horizontal_line_metrics()` для правильной установки baseline:

```rust
let text_y = if let Some(font) = self.font.as_ref() {
    if let Some(line_metrics) = font.horizontal_line_metrics(font_size) {
        cell_y + line_metrics.ascent.round() as usize
    } else {
        cell_y + 14 // fallback
    }
} else {
    cell_y + 14
};
```

### Проблема: Разные ymin для разных символов

**Это не проблема!** Разные `ymin` - это нормальное поведение шрифтов. При правильной baseline все символы выравниваются корректно.

**Не делать:**
- ❌ Нормализацию ymin
- ❌ Использование фиксированного ymin
- ❌ Специальные случаи для разных символов

**Правильно:**
- ✅ Использовать `baseline_y + metrics.ymin` как есть
- ✅ Правильно вычислять baseline через `horizontal_line_metrics()`

## Производительность

### Оптимизации

1. **Кэширование глифов:** `FontAtlas` кэширует растрированные глифы
2. **Округление размера:** размер шрифта округляется до 0.5 для уменьшения размера кэша
3. **Целочисленный блендинг:** используется быстрый целочисленный альфа-блендинг
4. **Проверка границ:** проверка границ буфера перед записью пикселей

### Ограничения

- Максимальная длина текста в bitmap fallback: 50 символов
- Максимальный масштаб изображения: 10x
- Размер кэша не ограничен (может расти при использовании многих символов)

## Примеры использования

### Отрисовка заголовка

```rust
// В draw_figure()
let title = &titles[idx];
let font_size = 20.0 * self.scale_factor;
let text_width = Self::calculate_text_width(self.font.as_ref(), title, font_size);
let text_x = (cell_x as f32 + (cell_width_usize as f32 - text_width) / 2.0).round() as usize;
let text_y = /* вычисление baseline */;

Self::draw_text_improved(
    &mut self.atlas,
    self.font.as_ref(),
    &mut buffer,
    text_x,
    text_y,
    title,
    font_size,
    self.scale_factor,
    buffer_width_usize,
    buffer_height_usize,
);
```

## Заключение

Система отрисовки текста в модуле plot использует современные подходы:

1. ✅ Baseline-ориентированную координатную модель
2. ✅ Кэширование глифов для оптимизации
3. ✅ DPI-aware рендеринг
4. ✅ Поддержку UTF-8 и кириллицы
5. ✅ Альфа-блендинг для сглаживания
6. ✅ Правильное использование метрик шрифта

Ключевой момент: правильное вычисление baseline через `horizontal_line_metrics()` обеспечивает корректное выравнивание всех символов независимо от их индивидуальных метрик.



