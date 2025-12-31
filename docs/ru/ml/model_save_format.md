# Формат сохранения моделей ML в DataCode

## Обзор

Модуль ML в DataCode сохраняет нейронные сети в бинарном формате с расширением `.nn`. Формат файла состоит из заголовка, JSON-метаданных архитектуры и бинарных данных тензоров (весов и смещений).

## Структура файла модели

Файл модели имеет следующую структуру:

```
[Заголовок]
├── Magic number: "DATACODE" (8 байт)
├── Version: 1 (4 байта, little-endian)
├── JSON length: длина JSON-строки (4 байта, little-endian)
│
[Метаданные архитектуры - JSON]
├── layers: массив слоёв
├── device: устройство вычислений ("cpu" или "metal")
└── training: информация о тренировке
│
[Бинарные данные тензоров]
├── Количество тензоров (4 байта)
└── Для каждого тензора:
    ├── Длина имени (4 байта)
    ├── Имя тензора (UTF-8)
    ├── Количество измерений shape (4 байта)
    ├── Размеры shape (каждое измерение - 4 байта, u32)
    └── Данные тензора (каждое значение f32 - 4 байта, little-endian)
```

## Сохраняемые данные

### 1. Архитектура сети (JSON)

#### Слои (layers)

Для каждого слоя сохраняется:

**Linear слои:**
```json
{
  "name": "layer0",
  "type": "Linear",
  "in_features": 784,
  "out_features": 128,
  "trainable": true
}
```

**Активационные слои:**
```json
{
  "name": "layer1",
  "type": "ReLU"  // или "Sigmoid", "Tanh", "Softmax", "Flatten"
}
```

#### Устройство (device)
- `"cpu"` - CPU вычисления
- `"metal"` - GPU вычисления (macOS Metal)
- `"cuda"` - GPU вычисления (NVIDIA CUDA на Linux/Windows)
- `"auto"` - Автоматический выбор устройства (GPU если доступен, иначе CPU)

### 2. Информация о тренировке (training)

#### Training Stages (stages)

Массив объектов `TrainingStage`, каждый содержит:

```json
{
  "epochs": 10,                    // Количество эпох в этой стадии
  "loss": "cross_entropy",         // Тип функции потерь: "cross_entropy", "sparse_cross_entropy", "categorical_cross_entropy", "binary_cross_entropy", "mse"
  "optimizer_type": "Adam",        // Тип оптимизатора: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW"
  "optimizer_params": {            // Параметры оптимизатора (сериализуются в JSON)
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  },
  "frozen_layers": [],            // Список замороженных слоёв
  "trainable_params": 101770,     // Количество обучаемых параметров
  "frozen_params": 0,             // Количество замороженных параметров
  "loss_history": [0.5, 0.3, ...], // История потерь по эпохам
  "accuracy_history": [0.75, ...], // История точности по эпохам
  "val_loss_history": [0.4, ...], // История валидационных потерь (опционально, может быть null)
  "val_accuracy_history": [0.8, ...] // История валидационной точности (опционально, может быть null)
}
```

**Важно**: Параметры оптимизатора (`optimizer_params`) сохраняются в формате JSON и включают все специфичные для оптимизатора параметры (например, `beta1`, `beta2` для Adam, `beta` для Momentum, `gamma` для RMSprop, `weight_decay` для AdamW и т.д.). Это позволяет точно восстановить состояние оптимизатора при загрузке модели.

#### Legacy поля (для обратной совместимости)

```json
{
  "epochs": 10,                    // Общее количество эпох
  "loss": "categorical_cross_entropy",
  "optimizer": "Adam",
  "loss_history": [...],
  "accuracy_history": [...],
  "val_loss_history": [...],
  "val_accuracy_history": [...]
}
```

### 3. Параметры модели (тензоры)

Для каждого Linear слоя сохраняются два тензора:

1. **Веса (weights)**
   - Имя: `"layer{N}.weight"`
   - Форма: `[in_features, out_features]`
   - Данные: массив `f32` значений весов

2. **Смещения (bias)**
   - Имя: `"layer{N}.bias"`
   - Форма: `[out_features]` или `[1, out_features]` (при загрузке автоматически преобразуется в `[1, out_features]` для корректного broadcasting)
   - Данные: массив `f32` значений смещений

**Важно:** 
- Все тензоры перед сохранением конвертируются на CPU, даже если модель тренировалась на GPU.
- Формат bias может быть как `[out_features]`, так и `[1, out_features]` — при загрузке оба формата автоматически преобразуются в `[1, out_features]` для обеспечения корректного broadcasting при вычислениях.

## Пример структуры JSON архитектуры

```json
{
  "device": "cpu",  // Может быть "cpu", "metal", "cuda" или "auto"
  "layers": [
    {
      "name": "layer0",
      "type": "Linear",
      "in_features": 784,
      "out_features": 128,
      "trainable": true
    },
    {
      "name": "layer1",
      "type": "ReLU"
    },
    {
      "name": "layer2",
      "type": "Linear",
      "in_features": 128,
      "out_features": 10,
      "trainable": true
    }
  ],
  "training": {
    "stages": [
      {
        "epochs": 1,
        "loss": "cross_entropy",  // Может быть "cross_entropy", "sparse_cross_entropy", "categorical_cross_entropy", "binary_cross_entropy", "mse"
        "optimizer_type": "Adam",
        "optimizer_params": {
          "lr": 0.001,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 1e-8
        },
        "frozen_layers": [],
        "trainable_params": 101770,
        "frozen_params": 0,
        "loss_history": [0.7523],
        "accuracy_history": [0.7523],
        "val_loss_history": null,
        "val_accuracy_history": null
      }
    ],
    "epochs": 1,
    "loss": "cross_entropy",  // Legacy поле для обратной совместимости
    "optimizer": "Adam",      // Legacy поле для обратной совместимости
    "loss_history": [0.7523],
    "accuracy_history": [0.7523],
    "val_loss_history": null,
    "val_accuracy_history": null
  }
}
```

## Поддерживаемые оптимизаторы и их параметры

### SGD
```json
{"lr": 0.01}
```

### Momentum
```json
{"lr": 0.01, "beta": 0.9}
```

### NAG (Nesterov Accelerated Gradient)
```json
{"lr": 0.01, "beta": 0.9}
```

### Adagrad
```json
{"lr": 0.01, "epsilon": 1e-8}
```

### RMSprop
```json
{"lr": 0.001, "gamma": 0.99, "epsilon": 1e-8}
```

### Adam
```json
{"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
```

### AdamW
```json
{"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.01}
```

## Поддерживаемые функции потерь

- `"cross_entropy"` - Sparse cross entropy для многоклассовой классификации (class indices [N,1])
- `"sparse_cross_entropy"` - Deprecated алиас для `cross_entropy` (сохраняется как `"sparse_cross_entropy"`)
- `"categorical_cross_entropy"` - Categorical cross entropy для многоклассовой классификации (one-hot [N,C])
- `"binary_cross_entropy"` - для бинарной классификации
- `"mse"` - Mean Squared Error для регрессии

**Примечание**: `"cross_entropy"` сохраняется в файле модели как `"cross_entropy"` (не как `"sparse_softmax_cross_entropy"`). При загрузке модели эти значения используются для восстановления истории обучения, но не влияют на выбор функции потерь при дальнейшем обучении.

## Размер файла модели

Размер файла модели зависит от:

1. **Размера архитектуры:**
   - Количество слоёв
   - Размеры слоёв (in_features × out_features)

2. **Количества тензоров:**
   - Для каждого Linear слоя: 2 тензора (weights + bias)
   - Размер тензора = произведение размеров shape × 4 байта (f32)

3. **Истории тренировки:**
   - Количество эпох
   - Количество стадий тренировки
   - Размер массивов loss_history и accuracy_history

**Пример расчёта:**
- Слой: 784 → 128
  - weights: 784 × 128 × 4 байта = 401,408 байт
  - bias: 128 × 4 байта = 512 байт
  - Итого: ~402 KB на слой

## Версионирование

Текущая версия формата: **1**

При загрузке модели проверяется:
- Magic number должен быть "DATACODE"
- Version должен быть 1 (поддержка других версий может быть добавлена в будущем)

## Использование

### Сохранение модели

```datacode
import ml

model = ml.nn([784, 128, 10])
# ... тренировка модели ...
ml.save_model(model, "my_model.nn")
```

### Загрузка модели

```datacode
import ml

model = ml.load("my_model.nn")
ml.model_info(model)  // Показать информацию о модели
```

## Примечания

1. **Устройство при загрузке:** Модели всегда загружаются на CPU, даже если были сохранены с GPU. Это сделано для избежания проблем с инициализацией Metal на некоторых системах.

2. **Замороженные слои:** Информация о замороженных слоях сохраняется в training stages, но при загрузке все слои будут trainable. Необходимо явно заморозить слои после загрузки, если требуется.

3. **История тренировки:** Полная история тренировки сохраняется, что позволяет анализировать процесс обучения после загрузки модели.

4. **Обратная совместимость:** Формат поддерживает legacy поля для совместимости со старыми версиями моделей.

