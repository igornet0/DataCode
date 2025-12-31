# Схема обучения нейронной сети: Полный путь данных через граф вычислений

## Обзор

Данный документ описывает полный путь данных при обучении модели через метод `model.train()` в DataCode. Рассматривается архитектура MLP для MNIST: `Input(784) → Linear(128) + ReLU → Linear(10)`.

## Быстрый старт

Минимальный пример обучения модели:

```datacode
import ml

# Загрузка данных
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices

# Создание модели
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Обучение (cross_entropy использует class indices [N,1])
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Ключевые моменты**:
- `y_train` имеет форму `[N, 1]` - это class indices для `cross_entropy`
- Для `categorical_cross_entropy` нужно использовать one-hot: `ml.onehot(y_train, 10)`
- Подробнее см. секцию [3.1.1](#311-примеры-использования-функций-потерь)

## Вызов обучения

**Сигнатура метода `train()`**:
```datacode
(loss_history, accuracy_history) = model.train(
    x,                    # Features tensor [batch_size, num_features]
    y,                    # Targets tensor [batch_size, num_targets] или [batch_size, num_classes]
    epochs,               # Количество эпох
    batch_size,           # Размер батча
    learning_rate,        # Скорость обучения
    loss_type,            # Тип функции потерь: "mse", "cross_entropy", "categorical_cross_entropy", "binary_cross_entropy"
    x_val,                # Опциональный валидационный features tensor (может быть null)
    y_val,                # Опциональный валидационный targets tensor (может быть null)
    optimizer             # Опциональный оптимизатор: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (по умолчанию "SGD")
)
```

**Возвращаемое значение**: Кортеж `(loss_history, accuracy_history)` — два массива значений потерь и точности по эпохам.

**Пример для cross_entropy (class indices [N,1])**:
```datacode
y_train = ml.dataset_targets(dataset)  # [N, 1]
(loss_history, accuracy_history) = model.train(
    x_train, y_train, 
    5, 64, 0.001, 
    "cross_entropy", 
    x_val, y_val,
    "Adam"  # Опциональный параметр оптимизатора
)
```

**Параметры обучения**:
- `epochs` — количество эпох
- `batch_size` — размер батча
- `learning_rate` — скорость обучения
- `loss_type` — функция потерь: `"cross_entropy"`, `"sparse_cross_entropy"` (deprecated алиас для `cross_entropy`), `"categorical_cross_entropy"`, `"mse"`, `"binary_cross_entropy"`
- `x_val` — опциональный валидационный features tensor (может быть `null`)
- `y_val` — опциональный валидационный targets tensor (может быть `null`)
- `optimizer` — опциональный оптимизатор (по умолчанию `"SGD"`). Подробнее см. секцию [5.3](#53-таблица-оптимизаторов)

Подробные примеры для всех типов loss см. в секции [3.1.1](#311-примеры-использования-функций-потерь).

### Метод train_sh() (с early stopping и LR scheduling)

Метод `train_sh()` предоставляет расширенные возможности обучения с автоматической остановкой (early stopping) и планированием скорости обучения (learning rate scheduling).

**Сигнатура метода `train_sh()`**:
```datacode
history = model.train_sh(
    x,                    # Features tensor [batch_size, num_features]
    y,                    # Targets tensor [batch_size, num_targets] или [batch_size, num_classes]
    epochs,               # Максимальное количество эпох
    batch_size,           # Размер батча
    learning_rate,        # Начальная скорость обучения (будет изменяться планировщиком)
    loss_type,            # Тип функции потерь: "mse", "cross_entropy", "categorical_cross_entropy", "binary_cross_entropy"
    optimizer,            # Опциональный оптимизатор: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW" (по умолчанию "SGD")
    monitor,              # Метрика для мониторинга: "loss", "val_loss", "acc", "val_acc"
    patience,             # Количество эпох ожидания перед уменьшением LR или остановкой
    min_delta,            # Минимальный процент улучшения (например, 1.0 означает 1%)
    restore_best,         # Восстанавливать ли лучшие веса в конце обучения
    x_val,                # Опциональный валидационный features tensor (обязателен если monitor начинается с "val_")
    y_val                 # Опциональный валидационный targets tensor (обязателен если monitor начинается с "val_")
)
```

**Возвращаемое значение**: Объект `TrainingHistorySH` с полями:
- `loss` — история потерь на обучающей выборке
- `val_loss` — история потерь на валидационной выборке (опционально)
- `acc` — история точности на обучающей выборке
- `val_acc` — история точности на валидационной выборке (опционально)
- `lr` — история изменения скорости обучения
- `best_metric` — лучшее значение метрики
- `best_epoch` — эпоха с лучшей метрикой
- `stopped_epoch` — эпоха, на которой остановилось обучение

**Пример использования**:
```datacode
history = model.train_sh(
    x_train, y_train,
    100,              # Максимум 100 эпох
    64,               # Размер батча
    0.001,            # Начальная скорость обучения
    "cross_entropy",
    "Adam",           # Оптимизатор
    "val_loss",       # Мониторить валидационную потерю
    10,               # Ждать 10 эпох без улучшения
    1.0,              # Минимальное улучшение 1%
    true,             # Восстановить лучшие веса
    x_val, y_val
)

# Доступ к истории обучения
print("Лучшая эпоха:", history.best_epoch)
print("Лучшая метрика:", history.best_metric)
print("Остановлено на эпохе:", history.stopped_epoch)
```

## Архитектура графа вычислений

### Структура модели
- **Sequential контейнер** содержит список слоев: `[Linear(784→128), ReLU, Linear(128→10)]`
- **Graph (вычислительный граф)** - направленный ациклический граф (DAG), где:
  - **Узлы (Nodes)** представляют операции или входные данные
  - **Ребра** представляют зависимости между операциями
  - **Параметры** (веса и смещения) хранятся как входные узлы с `requires_grad = true`

### Параметры модели
Для архитектуры MLP создаются следующие параметры:
1. **Linear Layer 1**:
   - `Linear1_Weight`: [784, 128] - матрица весов
   - `Linear1_Bias`: [1, 128] - вектор смещений
2. **Linear Layer 2**:
   - `Linear2_Weight`: [128, 10] - матрица весов
   - `Linear2_Bias`: [1, 10] - вектор смещений

## Полный цикл обучения (один батч)

### Этап 1: Подготовка данных

**Файл**: `src/lib/ml/model.rs:282-402`

1. **Перемещение данных на устройство** (CPU/GPU):
   ```rust
   x_batch: [batch_size, 784]  // входные данные батча
   // Для cross_entropy:
   y_batch: [batch_size, 1]    // class indices (0-9)
   // Для categorical_cross_entropy:
   y_batch: [batch_size, 10]   // one-hot метки батча
   ```

2. **Обнуление градиентов**:
   ```rust
   self.sequential.zero_grad()  // Очищает все градиенты в графе
   ```

### Этап 2: Forward Pass (Прямой проход)

**Файл**: `src/lib/ml/model.rs:425` → `src/lib/ml/layer.rs:626-720`

#### 2.1. Построение графа вычислений

**Sequential.forward()** строит граф, проходя через все слои:

```rust
// Создание входного узла
input_node_id = graph.add_input()  // Узел 0: Input [batch_size, 784]
```

#### 2.2. Проход через Layer 1: Linear(784 → 128)

**Файл**: `src/lib/ml/layer.rs:200-333`

**Linear.forward()** создает операции в графе:

1. **Проверка формы входных данных**:
   ```rust
   // Валидация: input должен быть [batch_size, 784]
   assert_eq!(input.shape, [batch_size, 784]);
   ```

2. **Инициализация параметров** (если первый раз):
   ```rust
   Linear1_Weight = graph.add_input()  // Узел 1: Linear1_Weight [784, 128]
   Linear1_Bias = graph.add_input()   // Узел 2: Linear1_Bias [1, 128]
   ```

3. **Матричное умножение**:
   ```rust
   Linear1_MatMul = graph.add_op(MatMul, [input_node, Linear1_Weight])
   // Узел 3: Linear1_MatMul [batch_size, 784] @ [784, 128] = [batch_size, 128]
   // Проверка формы: выход [batch_size, 128]
   ```

4. **Добавление смещения** (с broadcast):
   ```rust
   Linear1_Add = graph.add_op(Add, [Linear1_MatMul, Linear1_Bias])
   // Узел 4: Linear1_Add [batch_size, 128] + [1, 128] (broadcast) = [batch_size, 128]
   // Broadcast: [1, 128] автоматически расширяется до [batch_size, 128]
   // Проверка формы: выход [batch_size, 128]
   ```

**Путь данных**: `x_batch [batch_size, 784]` → `Linear1_MatMul` → `[batch_size, 128]` → `Linear1_Add` → `[batch_size, 128]`

#### 2.3. Проход через Layer 2: ReLU

**Файл**: `src/lib/ml/layer.rs:428-431`

```rust
// Проверка формы входных данных
assert_eq!(Linear1_Add.shape, [batch_size, 128]);

ReLU1 = graph.add_op(ReLU, [Linear1_Add])
// Узел 5: ReLU1 [batch_size, 128] → [batch_size, 128]
// ReLU(x) = max(0, x) - применяется поэлементно
// Форма сохраняется: [batch_size, 128]
```

**Путь данных**: `[batch_size, 128]` → `ReLU1` → `[batch_size, 128]` (отрицательные значения обнуляются)

#### 2.4. Проход через Layer 3: Linear(128 → 10)

**Файл**: `src/lib/ml/layer.rs:200-333`

1. **Проверка формы входных данных**:
   ```rust
   // Валидация: ReLU1 должен быть [batch_size, 128]
   assert_eq!(ReLU1.shape, [batch_size, 128]);
   ```

2. **Инициализация параметров**:
   ```rust
   Linear2_Weight = graph.add_input()  // Узел 6: Linear2_Weight [128, 10]
   Linear2_Bias = graph.add_input()    // Узел 7: Linear2_Bias [1, 10]
   ```

3. **Матричное умножение**:
   ```rust
   Linear2_MatMul = graph.add_op(MatMul, [ReLU1, Linear2_Weight])
   // Узел 8: Linear2_MatMul [batch_size, 128] @ [128, 10] = [batch_size, 10]
   // Проверка формы: выход [batch_size, 10]
   ```

4. **Добавление смещения** (с broadcast):
   ```rust
   Linear2_Add = graph.add_op(Add, [Linear2_MatMul, Linear2_Bias])
   // Узел 9: Linear2_Add [batch_size, 10] + [1, 10] (broadcast) = [batch_size, 10]
   // Broadcast: [1, 10] автоматически расширяется до [batch_size, 10]
   // Проверка формы: выход [batch_size, 10] (logits)
   ```

**Путь данных**: `[batch_size, 128]` → `Linear2_MatMul` → `[batch_size, 10]` → `Linear2_Add` → `[batch_size, 10]`

#### 2.5. Выполнение forward pass

**Файл**: `src/lib/ml/graph.rs:140-302`

**Graph.forward()** выполняет операции в топологическом порядке:

1. **Топологическая сортировка** (алгоритм Кана):
   ```rust
   execution_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   // Порядок: Input → Params → MatMul → Add → ReLU → MatMul → Add
   ```

2. **Выполнение операций**:
   - Узел 0 (Input): `value = x_batch`
   - Узел 1 (Linear1_Weight): `value = Linear1_Weight` (из кэша или инициализации)
   - Узел 2 (Linear1_Bias): `value = Linear1_Bias`
   - Узел 3 (Linear1_MatMul): `value = input @ Linear1_Weight`
   - Узел 4 (Linear1_Add): `value = Linear1_MatMul + Linear1_Bias` (broadcast)
   - Узел 5 (ReLU1): `value = relu(Linear1_Add)`
   - Узел 6 (Linear2_Weight): `value = Linear2_Weight`
   - Узел 7 (Linear2_Bias): `value = Linear2_Bias`
   - Узел 8 (Linear2_MatMul): `value = ReLU1 @ Linear2_Weight`
   - Узел 9 (Linear2_Add): `value = Linear2_MatMul + Linear2_Bias` (broadcast)

**Результат**: `logits [batch_size, 10]` - логиты (сырые предсказания)

### Этап 3: Вычисление функции потерь

**Файл**: `src/lib/ml/model.rs:514-662`

#### 3.1. Таблица функций потерь

| Loss name | Targets format | Shape | Description |
|-----------|----------------|-------|-------------|
| `cross_entropy` | class indices | [N,1] | Sparse cross entropy with class indices (int) |
| `sparse_cross_entropy` | class indices | [N,1] | **Deprecated**: алиас для `cross_entropy`. Используйте `cross_entropy` |
| `categorical_cross_entropy` | one-hot | [N,C] | Cross entropy with one-hot encoding |
| `binary_cross_entropy` | binary | [N,1] | Binary classification (values in [0,1]) |
| `mse` | continuous | [N,C] | Mean squared error for regression |

**Важно**: 
- `cross_entropy` ожидает **индексы классов** [N,1], а не one-hot
- `sparse_cross_entropy` — deprecated алиас для `cross_entropy`, рекомендуется использовать `cross_entropy`
- `categorical_cross_entropy` ожидает **one-hot** [N,C]
- Формат проверяется строго, ошибки формата приводят к ошибке компиляции/выполнения

#### 3.1.1. Примеры использования функций потерь

**Правильно: cross_entropy с class indices [N,1]**

```datacode
# Загрузка данных
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices (0-9)

# Обучение с cross_entropy
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Правильно: categorical_cross_entropy с one-hot [N,C]**

```datacode
# Загрузка данных
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)    # [60000, 1] - class indices

# Конвертация в one-hot
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10] - one-hot encoding

# Обучение с categorical_cross_entropy
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

**НЕПРАВИЛЬНО: несоответствие формата и loss_type**

```datacode
# ❌ ОШИБКА: cross_entropy с one-hot
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val_onehot)
# Ошибка: "cross_entropy expects class indices [batch, 1], got [batch, 10]"

# ❌ ОШИБКА: categorical_cross_entropy с class indices
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val)
# Ошибка: "categorical_cross_entropy expects one-hot targets [batch, 10], got [batch, 1]"
```

При неправильном формате система выдаст понятное сообщение об ошибке с подсказкой, как исправить проблему.

#### 3.1.2. Таблица совместимости loss_type и accuracy_function

При вычислении точности (accuracy) важно использовать правильную функцию в зависимости от типа loss:

| loss_type | accuracy_function | targets format | Описание |
|-----------|-------------------|----------------|----------|
| `cross_entropy` | `compute_accuracy_sparse` | [N,1] class indices | Сравнивает argmax(logits) с class indices |
| `sparse_cross_entropy` | `compute_accuracy_sparse` | [N,1] class indices | Deprecated алиас для `cross_entropy` |
| `categorical_cross_entropy` | `compute_accuracy_categorical` | [N,C] one-hot | Сравнивает argmax(logits) с argmax(one-hot) |
| `binary_cross_entropy` | N/A | [N,1] binary | Accuracy не вычисляется автоматически |
| `mse` | N/A | [N,C] continuous | Accuracy не применима для регрессии |

**Примечание**: Функции accuracy вызываются автоматически внутри `model.train()` на основе `loss_type`. Для `cross_entropy` и `sparse_cross_entropy` используется `compute_accuracy_sparse()`, для `categorical_cross_entropy` - `compute_accuracy_categorical()`.

#### 3.2. Добавление целевых значений в граф

```rust
target_node_id = graph.add_input()  // Узел 10: Targets
graph.nodes[target_node_id].value = Some(y_batch)
```

#### 3.3. Создание узла потерь

Для `loss_type = "cross_entropy"` (sparse, class indices [N,1]):

```rust
loss_node = graph.add_op(CrossEntropy, [output_node_id, target_node_id])
// Узел 11: CrossEntropy [batch_size, 10], [batch_size, 1] → [1] (скаляр)
```

Для `loss_type = "categorical_cross_entropy"` (one-hot [N,C]):

```rust
loss_node = graph.add_op(CategoricalCrossEntropy, [output_node_id, target_node_id])
// Узел 11: CategoricalCrossEntropy [batch_size, 10], [batch_size, 10] → [1] (скаляр)
```

**Вычисление CrossEntropy (sparse)** (`src/lib/ml/graph.rs:267-283`):
1. Применяется **Softmax** к logits (внутри операции для численной стабильности)
2. Вычисляется **Cross-Entropy**: `loss = -mean(log(softmax(logits)[target_class]))`
3. Результат: скалярное значение потерь `[1]`

**Вычисление CategoricalCrossEntropy** (`src/lib/ml/graph.rs`):
1. Применяется **Softmax** к logits (внутри операции для численной стабильности)
2. Вычисляется **Cross-Entropy**: `loss = -mean(sum(targets * log(softmax(logits))))`
3. Результат: скалярное значение потерь `[1]`

**Путь данных для cross_entropy**: 
- `logits [batch_size, 10]` + `targets [batch_size, 1]` (class indices)
- → `CrossEntropy` 
- → `loss [1]` (скаляр)

**Путь данных для categorical_cross_entropy**: 
- `logits [batch_size, 10]` + `targets [batch_size, 10]` (one-hot)
- → `CategoricalCrossEntropy` 
- → `loss [1]` (скаляр)

### Этап 4: Backward Pass (Обратный проход)

**Файл**: `src/lib/ml/model.rs:712` → `src/lib/ml/graph.rs:359-750`

#### 4.0. Инициализация градиентов (zero_grad)

**Важно**: Перед каждым backward pass все градиенты должны быть обнулены. Это происходит автоматически в Этапе 1 (Подготовка данных) через вызов `self.sequential.zero_grad()`.

```rust
// Вызывается в начале каждой итерации батча (Этап 1)
self.sequential.zero_grad()
// Очищает все градиенты в графе, включая градиенты параметров
// Это критически важно, так как градиенты накапливаются между итерациями
```

**Почему это важно**:
- Градиенты накапливаются для параметров между батчами
- Без `zero_grad()` градиенты будут суммироваться, что приведет к неправильному обучению
- `zero_grad()` сбрасывает все градиенты перед новым forward pass
- Градиенты вычисляются заново для каждого батча

#### 4.1. Инициализация градиента потерь

```rust
graph.nodes[loss_node].grad = Tensor::ones([1])  // grad_loss = 1.0
```

#### 4.2. Обратное распространение градиентов

**Graph.backward()** проходит по графу в **обратном топологическом порядке**:

```rust
backward_order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
// Обратный порядок: Loss → Output → Linear2_Add → Linear2_MatMul → ReLU1 → Linear1_Add → Linear1_MatMul → Params
```

#### 4.3. Вычисление градиентов для каждой операции

**Узел 11: CrossEntropy** (`src/lib/ml/graph.rs:714-749`)

```rust
// Градиент по logits: (softmax(logits) - targets) / batch_size
grad_logits = (softmax(logits) - targets) / batch_size
// Форма: [batch_size, 10]
```

**Путь градиента**: `loss [1]` → `CrossEntropy.backward()` → `grad_logits [batch_size, 10]`

---

**Узел 9: Linear2_Add (Linear2_Bias)** (`src/lib/ml/graph.rs:411-429`)

```rust
// Градиент распространяется на оба входа:
grad_Linear2_MatMul = grad_Linear2_Add.clone()  // [batch_size, 10]
grad_Linear2_Bias = grad_Linear2_Add.sum_to_shape([1, 10])  // Суммирование по batch
```

**Путь градиента**: `grad_Linear2_Add [batch_size, 10]` → `Linear2_Add.backward()` → 
- `grad_Linear2_MatMul [batch_size, 10]`
- `grad_Linear2_Bias [1, 10]` (сохраняется в узле 7: Linear2_Bias)

---

**Узел 8: Linear2_MatMul (Linear2_Weight)** (`src/lib/ml/graph.rs:490-591`)

```rust
// Для y = a @ b, где a=[batch, 128], b=[128, 10]
// grad_a = grad_y @ b^T
// grad_b = a^T @ grad_y

grad_ReLU1 = grad_Linear2_MatMul @ Linear2_Weight^T
// [batch_size, 10] @ [10, 128] = [batch_size, 128]

grad_Linear2_Weight = ReLU1^T @ grad_Linear2_MatMul
// [128, batch_size] @ [batch_size, 10] = [128, 10]
```

**Путь градиента**: `grad_Linear2_MatMul [batch_size, 10]` → `Linear2_MatMul.backward()` →
- `grad_ReLU1 [batch_size, 128]` (к узлу 5: ReLU1)
- `grad_Linear2_Weight [128, 10]` (сохраняется в узле 6: Linear2_Weight)

---

**Узел 5: ReLU1** (`src/lib/ml/graph.rs:625-638`)

```rust
// Градиент ReLU: grad = grad * (x > 0)
// Если вход был > 0, градиент проходит, иначе = 0

grad_Linear1_Add = grad_ReLU1 * mask
// mask[i] = 1.0 если Linear1_Add[i] > 0, иначе 0.0
// Форма: [batch_size, 128]
```

**Путь градиента**: `grad_ReLU1 [batch_size, 128]` → `ReLU1.backward()` → 
- `grad_Linear1_Add [batch_size, 128]` (только для положительных значений)

---

**Узел 4: Linear1_Add (Linear1_Bias)** (`src/lib/ml/graph.rs:411-429`)

```rust
grad_Linear1_MatMul = grad_Linear1_Add.clone()  // [batch_size, 128]
grad_Linear1_Bias = grad_Linear1_Add.sum_to_shape([1, 128])  // Суммирование по batch
```

**Путь градиента**: `grad_Linear1_Add [batch_size, 128]` → `Linear1_Add.backward()` →
- `grad_Linear1_MatMul [batch_size, 128]`
- `grad_Linear1_Bias [1, 128]` (сохраняется в узле 2: Linear1_Bias)

---

**Узел 3: Linear1_MatMul (Linear1_Weight)** (`src/lib/ml/graph.rs:490-591`)

```rust
grad_input = grad_Linear1_MatMul @ Linear1_Weight^T
// [batch_size, 128] @ [128, 784] = [batch_size, 784]

grad_Linear1_Weight = input^T @ grad_Linear1_MatMul
// [784, batch_size] @ [batch_size, 128] = [784, 128]
```

**Путь градиента**: `grad_Linear1_MatMul [batch_size, 128]` → `Linear1_MatMul.backward()` →
- `grad_input [batch_size, 784]` (к узлу 0: Input, не используется)
- `grad_Linear1_Weight [784, 128]` (сохраняется в узле 1: Linear1_Weight)

---

#### 4.4. Итоговые градиенты параметров

После backward pass градиенты сохранены в узлах параметров:

- **Узел 1 (Linear1_Weight)**: `grad_Linear1_Weight [784, 128]`
- **Узел 2 (Linear1_Bias)**: `grad_Linear1_Bias [1, 128]`
- **Узел 6 (Linear2_Weight)**: `grad_Linear2_Weight [128, 10]`
- **Узел 7 (Linear2_Bias)**: `grad_Linear2_Bias [1, 10]`

**Важно**: Градиенты накапливаются для параметров. Функция `zero_grad()` вызывается в начале каждой итерации (Этап 1), чтобы сбросить все градиенты перед новым forward pass. Градиенты вычисляются заново для каждого батча.

### Этап 5: Обновление параметров (Optimizer Step)

**Файл**: `src/lib/ml/model.rs:836` → `src/lib/ml/optimizer.rs:36-95`

#### 5.1. SGD Optimizer

**SGD.step()** обновляет каждый параметр:

```rust
for each param_node_id in [1, 2, 6, 7]:
    current_value = graph.get_output(param_node_id)  // Текущее значение параметра
    gradient = graph.get_gradient(param_node_id)      // Градиент параметра
    
    // Обновление: new_value = current_value - lr * gradient
    update = lr * gradient
    new_value = current_value - update
    
    graph.nodes[param_node_id].value = Some(new_value)  // Сохранить новое значение
```

**Пример для Linear1_Weight**:
```rust
Linear1_Weight_new = Linear1_Weight_old - learning_rate * grad_Linear1_Weight
// [784, 128] = [784, 128] - lr * [784, 128]
```

#### 5.2. Сохранение обновленных параметров

```rust
self.sequential.save_parameter_values()
// Сохраняет обновленные значения параметров в кэш для следующего forward pass
```

#### 5.3. Таблица оптимизаторов

DataCode поддерживает различные оптимизаторы для обновления параметров модели. Все оптимизаторы работают с одним и тем же графом вычислений и используют градиенты, вычисленные в backward pass. Разница между оптимизаторами заключается только в формуле обновления параметров.

**Общие принципы**:
- Все оптимизаторы обновляют параметры **in-place** (напрямую изменяют значения в графе)
- Работают через `optimizer.step(graph, param_node_ids)`
- Выбор оптимизатора не влияет на структуру графа или backward pass
- Градиенты вычисляются одинаково для всех оптимизаторов

##### 5.3.1. Таблица оптимизаторов

| Оптимизатор | Формула обновления весов | Параметры | Пример использования |
|-------------|-------------------------|-----------|---------------------|
| **SGD** | `w = w - η * grad` | `η` — learning rate | `optimizer = "SGD"` или `optimizer = SGD(lr=0.01)` |
| **SGD + Momentum** | `v = β * v + (1-β) * grad`<br>`w = w - η * v` | `η` — lr, `β` — momentum (0.9) | `optimizer = "Momentum"` или `optimizer = Momentum(lr=0.01, beta=0.9)` |
| **Nesterov (NAG)** | `v = β * v + η * grad(w - β*v)`<br>`w = w - v` | `η` — lr, `β` — momentum | `optimizer = "NAG"` или `optimizer = NAG(lr=0.01, beta=0.9)` |
| **Adagrad** | `G = G + grad²`<br>`w = w - η / sqrt(G + ε) * grad` | `η` — lr, `ε` — малое число (1e-8) | `optimizer = "Adagrad"` или `optimizer = Adagrad(lr=0.01)` |
| **RMSprop** | `E[g²] = γ * E[g²] + (1-γ) * grad²`<br>`w = w - η / sqrt(E[g²] + ε) * grad` | `η` — lr, `γ` — decay (0.9–0.99), `ε` — малое число | `optimizer = "RMSprop"` или `optimizer = RMSprop(lr=0.001, gamma=0.9)` |
| **Adam** | `m = β1*m + (1-β1)*grad`<br>`v = β2*v + (1-β2)*grad²`<br>`m̂ = m / (1-β1^t)`<br>`v̂ = v / (1-β2^t)`<br>`w = w - η * m̂ / (sqrt(v̂) + ε)` | `η` — lr, `β1=0.9`, `β2=0.999`, `ε=1e-8` | `optimizer = "Adam"` или `optimizer = Adam(lr=0.001)` |
| **AdamW** | `m = β1*m + (1-β1)*grad`<br>`v = β2*v + (1-β2)*grad²`<br>`m̂ = m / (1-β1^t)`<br>`v̂ = v / (1-β2^t)`<br>`w = w - η * (m̂ / (sqrt(v̂)+ε) + λ*w)` | `η` — lr, `β1=0.9`, `β2=0.999`, `λ` — weight decay | `optimizer = "AdamW"` или `optimizer = AdamW(lr=0.001, weight_decay=0.01)` |

**Обозначения в формулах**:
- `w` — веса (параметры модели)
- `grad` — градиент параметра
- `η` (eta) — learning rate (скорость обучения)
- `β` (beta) — коэффициент момента (momentum)
- `γ` (gamma) — коэффициент затухания (decay rate)
- `ε` (epsilon) — малое число для численной стабильности
- `λ` (lambda) — коэффициент weight decay
- `t` — номер итерации (time step)
- `m` — оценка первого момента (momentum)
- `v` — оценка второго момента (velocity)
- `m̂`, `v̂` — скорректированные моменты (bias-corrected)

##### 5.3.2. Примеры использования оптимизаторов

**Универсальный интерфейс в DataCode**:

```datacode
import ml

# Загрузка данных
dataset = ml.load_mnist("train")
x_train = ml.dataset_features(dataset)  # [60000, 784]
y_train = ml.dataset_targets(dataset)   # [60000, 1] - class indices

# Создание модели
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model_seq = ml.sequential([layer1, layer2, layer3])
model = ml.neural_network(model_seq)

# Настройки обучения
epochs = 10
batch_size = 64
learning_rate = 0.001
optimizer_name = "Adam"        # Выбираем оптимизатор: "SGD", "Momentum", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW"
loss_fn = "cross_entropy"

# Подготовка валидационных данных
x_val = ml.dataset_features(ml.load_mnist("test"))
y_val = ml.dataset_targets(ml.load_mnist("test"))

# Обучение модели
(loss_history, accuracy_history) = model.train(
    x_train, y_train,
    epochs,
    batch_size,
    learning_rate,
    loss_fn,
    x_val,
    y_val,
    optimizer_name  # Опциональный параметр
)
```

**Примеры для разных оптимизаторов**:

```datacode
# SGD (Stochastic Gradient Descent)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.01, 
                                                "cross_entropy", x_val, y_val, "SGD")

# Adam (рекомендуется для большинства задач)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "Adam")

# RMSprop (хорош для RNN)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "RMSprop")

# AdamW (с weight decay для регуляризации)
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val, "AdamW")
```

##### 5.3.3. Рекомендации по выбору оптимизатора

- **SGD**: Простой и надежный, хорошо работает для простых задач. Требует тщательного подбора learning rate.
- **SGD + Momentum**: Ускоряет сходимость SGD, помогает преодолевать локальные минимумы.
- **Nesterov (NAG)**: Улучшенная версия Momentum, часто сходится быстрее.
- **Adagrad**: Адаптивный learning rate, хорошо работает для разреженных градиентов. Может слишком сильно уменьшать learning rate.
- **RMSprop**: Решает проблему Adagrad с затухающим learning rate. Хорош для RNN.
- **Adam**: Рекомендуется для большинства задач. Адаптивный learning rate с моментом. Хорошо работает из коробки.
- **AdamW**: Улучшенная версия Adam с правильной реализацией weight decay. Рекомендуется для современных моделей.

**Важно**: Все оптимизаторы работают с одним и тем же графом вычислений и используют градиенты, вычисленные в backward pass. Разница только в формуле обновления параметров.

### Этап 6: Очистка графа

**Файл**: `src/lib/ml/model.rs:856` → `src/lib/ml/layer.rs:767-820`

```rust
self.sequential.clear_non_parameter_nodes()
// Удаляет все промежуточные узлы (MatMul, Add, ReLU, Loss)
// Сохраняет только узлы параметров (weights, biases)
// Это предотвращает утечки памяти между батчами
```

## Визуализация графа вычислений

### Параметры модели (сохраняются между батчами)

```
Параметры (зеленые узлы - сохраняются):
  Linear1_Weight [784, 128]  ──┐
  Linear1_Bias [1, 128]        │
  Linear2_Weight [128, 10]     │─── Обновляются через backward pass
  Linear2_Bias [1, 10]         │
                               └─── Сохраняются после clear_non_parameter_nodes()
```

### Forward Pass (Прямой проход) - синие стрелки

```
Input [batch, 784]  (синяя стрелка →)
  │
  ├─→ Linear1_Weight [784, 128] ──┐
  │                               │
  └─→ Linear1_MatMul ─────────────┼─→ [batch, 128]  (синяя стрелка →)
                                  │
  Linear1_Bias [1, 128] ──────────┘
                                  │
                                  ↓  (синяя стрелка →)
                        Linear1_Add [batch, 128]
                                  │
                                  ↓  (синяя стрелка →)
                        ReLU1 [batch, 128]
                                  │
                                  ├─→ Linear2_Weight [128, 10] ──┐
                                  │                              │
                                  └─→ Linear2_MatMul ────────────┼─→ [batch, 10]  (синяя стрелка →)
                                                                  │
                        Linear2_Bias [1, 10] ────────────────────┘
                                                                  │
                                                                  ↓  (синяя стрелка →)
                                                Linear2_Add [batch, 10] (logits)
                                                                  │
                                                                  ↓  (синяя стрелка →)
                                                CrossEntropy
                                                                  │
                                                                  ↓  (синяя стрелка →)
                                                 Loss [1]
```

### Backward Pass (Обратный проход) - красные стрелки

```
Loss [1] (grad = 1.0)  (красная стрелка ←)
  │
  ↓  (красная стрелка ←)
CrossEntropy.backward()
  │
  ↓ grad_logits [batch, 10]  (красная стрелка ←)
Linear2_Add.backward()
  │
  ├─→ grad_Linear2_MatMul [batch, 10]  (красная стрелка ←)
  └─→ grad_Linear2_Bias [1, 10] ──→ Linear2_Bias обновляется  (красная стрелка ←)
  │
  ↓  (красная стрелка ←)
Linear2_MatMul.backward()
  │
  ├─→ grad_ReLU1 [batch, 128]  (красная стрелка ←)
  └─→ grad_Linear2_Weight [128, 10] ──→ Linear2_Weight обновляется  (красная стрелка ←)
  │
  ↓  (красная стрелка ←)
ReLU1.backward() (mask: x > 0)
  │
  ↓ grad_Linear1_Add [batch, 128] (только положительные)  (красная стрелка ←)
Linear1_Add.backward()
  │
  ├─→ grad_Linear1_MatMul [batch, 128]  (красная стрелка ←)
  └─→ grad_Linear1_Bias [1, 128] ──→ Linear1_Bias обновляется  (красная стрелка ←)
  │
  ↓  (красная стрелка ←)
Linear1_MatMul.backward()
  │
  ├─→ grad_input [batch, 784] (не используется)  (красная стрелка ←)
  └─→ grad_Linear1_Weight [784, 128] ──→ Linear1_Weight обновляется  (красная стрелка ←)
```

**Легенда**:
- **Синие стрелки (→)**: Forward pass - поток данных от входов к выходу
- **Красные стрелки (←)**: Backward pass - поток градиентов от loss к параметрам
- **Зеленые узлы**: Параметры - сохраняются между батчами, обновляются через optimizer

## Полный цикл для одного батча

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Подготовка: x_batch, y_batch, zero_grad()                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Forward Pass:                                             │
│    x → Linear_1 → ReLU → Linear_2 → logits                   │
│    logits + targets → CrossEntropy → loss                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Backward Pass:                                            │
│    loss → grad_logits → grad_weight_2, grad_bias_2           │
│    → grad_relu → grad_weight_1, grad_bias_1                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Optimizer Step:                                            │
│    weight_1 = weight_1 - lr * grad_weight_1                  │
│    bias_1 = bias_1 - lr * grad_bias_1                        │
│    weight_2 = weight_2 - lr * grad_weight_2                  │
│    bias_2 = bias_2 - lr * grad_bias_2                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Сохранение и очистка:                                      │
│    save_parameter_values()                                   │
│    clear_non_parameter_nodes()                               │
└─────────────────────────────────────────────────────────────┘
```

## Ключевые файлы и функции

### Основные компоненты

1. **model.rs** (`src/lib/ml/model.rs`):
   - `NeuralNetwork::train()` - главный цикл обучения
   - `NeuralNetwork::forward()` - вызов forward pass

2. **layer.rs** (`src/lib/ml/layer.rs`):
   - `Sequential::forward()` - построение и выполнение графа
   - `Linear::forward()` - создание операций Linear слоя
   - `ReLU::forward()` - создание операции ReLU

3. **graph.rs** (`src/lib/ml/graph.rs`):
   - `Graph::forward()` - выполнение операций в топологическом порядке
   - `Graph::backward()` - обратное распространение градиентов
   - `Graph::topological_sort()` - определение порядка выполнения

4. **optimizer.rs** (`src/lib/ml/optimizer.rs`):
   - `SGD::step()` - обновление параметров по градиентам (SGD)
   - `Adam::step()` - обновление параметров по градиентам (Adam)
   - Поддержка других оптимизаторов: Momentum, NAG, Adagrad, RMSprop, AdamW
   - Подробнее см. секцию [5.3](#53-таблица-оптимизаторов)

5. **loss.rs** (`src/lib/ml/loss.rs`):
   - `softmax_cross_entropy_loss()` - вычисление функции потерь

## Особенности реализации

### 1. Автоматическое дифференцирование (Autograd)

**Основные возможности**:
- Граф вычислений строится **динамически** во время forward pass
- Градиенты вычисляются автоматически через backward pass
- Не требуется ручное вычисление производных

**Расширенные возможности**:

1. **Ветвления (Branching)**:
   - Граф может иметь несколько путей вычислений
   - Backward pass корректно обрабатывает все ветви
   - Градиенты суммируются в точках слияния

2. **Условные конструкции**:
   - Поддержка `if/else` в графе вычислений
   - Градиенты распространяются только по активной ветке
   - Условные операции интегрированы в autograd

3. **Циклы**:
   - Поддержка `for/while` циклов в графе
   - Градиенты накапливаются через итерации цикла
   - Backward pass проходит через все итерации

4. **Динамическая структура**:
   - Граф перестраивается на каждом forward pass
   - Это позволяет изменять архитектуру между итерациями
   - Поддерживаются модели с переменной структурой

**Преимущества**:
- Гибкость: можно строить сложные архитектуры
- Удобство: не нужно вручную вычислять градиенты
- Безопасность: система гарантирует корректность вычислений

### 2. Управление памятью
- Промежуточные узлы удаляются после каждого батча
- Сохраняются только параметры (weights, biases)
- Параметры кэшируются для следующего forward pass

### 3. Поддержка устройств (CPU/GPU)
- Все операции поддерживают работу на CPU и GPU
- Данные автоматически перемещаются на нужное устройство
- Градиенты сохраняются на том же устройстве, что и параметры

### 5. Гибкость batch_size

**Поддержка различных размеров батча**:
- `batch_size` может быть **1** (online learning / stochastic gradient descent)
- `batch_size` может быть равен размеру датасета (full batch gradient descent)
- `batch_size` может быть любым значением между ними (mini-batch)

**Как это работает**:
- Топологическая сортировка работает для **любого** batch_size
- Backward pass работает корректно для любого batch_size
- Градиенты автоматически нормализуются на batch_size
- Формула: `gradient = gradient / batch_size` (внутри loss функций)

**Примеры**:
```datacode
# Online learning (batch_size = 1)
model.train(x_train, y_train, 10, 1, 0.001, "cross_entropy", ...)

# Mini-batch (batch_size = 64)
model.train(x_train, y_train, 10, 64, 0.001, "cross_entropy", ...)

# Full batch (batch_size = dataset_size)
model.train(x_train, y_train, 10, x_train.shape[0], 0.001, "cross_entropy", ...)
```

**Производительность**:
- Меньший batch_size → больше итераций, но меньше памяти
- Больший batch_size → меньше итераций, но больше памяти
- Оптимальный batch_size зависит от размера модели и доступной памяти

### 4. Численная стабильность

**Softmax и CrossEntropy**:
- CrossEntropy и CategoricalCrossEntropy используют **fused Softmax** для избежания переполнения
- Применяется **log-sum-exp трюк** для численной стабильности:
  ```rust
  // Вместо прямого вычисления softmax:
  // softmax(x) = exp(x) / sum(exp(x))  // Может переполниться!
  
  // Используется стабильная формула:
  max_x = max(x)
  log_softmax(x) = x - max_x - log(sum(exp(x - max_x)))
  ```
- Это предотвращает переполнение при больших значениях logits
- Формула: `log_softmax(x) = x - log(sum(exp(x - max(x))))`
- Все операции loss используют этот трюк автоматически

**ReLU**:
- ReLU предотвращает распространение градиентов через отрицательные значения
- Градиент = 0 для отрицательных входов, что стабилизирует обучение

**Другие меры**:
- Все операции проверяют на NaN/Inf
- Градиенты нормализуются на batch_size автоматически
- Используются epsilon значения для предотвращения деления на ноль

## Частые ошибки и их решения

### 1. Несоответствие формата targets и loss_type

**Ошибка**: Использование one-hot с `cross_entropy`
```datacode
# ❌ НЕПРАВИЛЬНО
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val_onehot)
# Ошибка: "cross_entropy expects class indices [batch, 1], got [batch, 10]"
```

**Решение**: Использовать class indices для `cross_entropy`
```datacode
# ✅ ПРАВИЛЬНО
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "cross_entropy", x_val, y_val)
```

**Или**: Использовать `categorical_cross_entropy` с one-hot
```datacode
# ✅ ПРАВИЛЬНО
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

### 2. Использование class indices с categorical_cross_entropy

**Ошибка**:
```datacode
# ❌ НЕПРАВИЛЬНО
y_train = ml.dataset_targets(dataset)  # [60000, 1]
(loss_history, accuracy_history) = model.train(x_train, y_train, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val)
# Ошибка: "categorical_cross_entropy expects one-hot targets [batch, 10], got [batch, 1]"
```

**Решение**: Конвертировать в one-hot
```datacode
# ✅ ПРАВИЛЬНО
y_train_onehot = ml.onehot(y_train, 10)  # [60000, 10]
(loss_history, accuracy_history) = model.train(x_train, y_train_onehot, 
                                                5, 64, 0.001, 
                                                "categorical_cross_entropy", x_val, y_val_onehot)
```

### 3. Забыть zero_grad() (не применимо - вызывается автоматически)

**Примечание**: В DataCode `zero_grad()` вызывается автоматически в начале каждой итерации батча (Этап 1). Не нужно вызывать вручную. Если градиенты накапливаются неправильно, это может быть баг в системе, а не ошибка пользователя.

### 4. Неправильный формат targets для accuracy

**Проблема**: Accuracy вычисляется автоматически на основе `loss_type`. Если используется неправильный формат, accuracy может быть вычислена неправильно.

**Решение**: Убедиться, что формат targets соответствует `loss_type`:
- `cross_entropy` → `[N, 1]` class indices → `compute_accuracy_sparse()`
- `categorical_cross_entropy` → `[N, C]` one-hot → `compute_accuracy_categorical()`

### 5. Несоответствие размеров батча

**Ошибка**: Разные размеры батчей для train и validation
```datacode
# Может вызвать проблемы, если batch_size не делит размер датасета
```

**Решение**: Система автоматически обрабатывает последний неполный батч. Но лучше использовать batch_size, который делит размер датасета.

## Заключение

Обучение модели в DataCode происходит через построение динамического вычислительного графа, который:
1. Выполняет forward pass для получения предсказаний
2. Вычисляет функцию потерь
3. Распространяет градиенты обратно через граф (backward pass)
4. Обновляет параметры с помощью оптимизатора
5. Очищает промежуточные узлы для следующего батча

Этот подход обеспечивает гибкость, автоматическое дифференцирование и эффективное использование памяти.

**Ключевые принципы**:
- ✅ `cross_entropy` использует class indices `[N, 1]`
- ✅ `categorical_cross_entropy` использует one-hot `[N, C]`
- ✅ Формат проверяется строго с понятными сообщениями об ошибках
- ✅ Градиенты обнуляются автоматически перед каждым батчем
- ✅ Память управляется автоматически через очистку промежуточных узлов

