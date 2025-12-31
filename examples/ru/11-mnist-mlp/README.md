# Пример MNIST MLP

Этот пример демонстрирует обучение минимального многослойного перцептрона (MLP) на наборе данных MNIST для классификации цифр.

## Архитектура

Модель использует простую архитектуру MLP:

```
Вход: [batch, 784]
  ↓
Linear(784, 128) + инициализация He
  ↓
ReLU
  ↓
Linear(128, 10)
  ↓
Softmax
  ↓
SparseCrossEntropyLoss
  ↓
Оптимизатор SGD (lr=0.001)
```

## Пример кода

Пример демонстрирует новый API слоёв с использованием `ml.layer.*`:

```datacode
import ml

# Создание слоёв с использованием нового API
layer1 = ml.layer.linear(784, 128)  # Полносвязный слой: 784 → 128
layer2 = ml.layer.relu()            # Активация ReLU
layer3 = ml.layer.linear(128, 10)   # Полносвязный слой: 128 → 10
layer4 = ml.layer.softmax()         # Активация Softmax

# Создание контейнера Sequential
layers = [layer1, layer2, layer3, layer4]
model_seq = ml.sequential(layers)

# Создание нейронной сети
model = ml.neural_network(model_seq)

# Загрузка и подготовка данных
dataset_train = ml.load_mnist("train")
x_train = ml.dataset_features(dataset_train)
y_train = ml.dataset_targets(dataset_train)

# Обучение модели
loss_history = ml.nn_train(model, x_train, y_train, epochs=5, batch_size=32, learning_rate=0.001, loss="sparse_cross_entropy")
```

## Ожидаемые результаты

- **Точность**: 92-95% на тестовом наборе после 5-10 эпох
- **Время обучения**: Зависит от оборудования, обычно несколько минут на CPU, значительно быстрее на GPU

## Использование

### Базовое использование (CPU)
```bash
# Если установлен глобально
datacode mnist_mlp.dc

# Или в режиме разработки
cargo run -- examples/ru/11-mnist-mlp/mnist_mlp.dc
```

### С ускорением GPU

**macOS (Metal):**
```bash
# Режим разработки с поддержкой Metal GPU
cargo run --features metal -- examples/ru/11-mnist-mlp/mnist_mlp.dc

# Или используя Makefile
make run-metal FILE=examples/ru/11-mnist-mlp/mnist_mlp.dc
```

**Linux/Windows (CUDA):**
```bash
# Режим разработки с поддержкой CUDA GPU
cargo run --features cuda -- examples/ru/11-mnist-mlp/mnist_mlp.dc

# Или используя Makefile
make run-cuda FILE=examples/ru/11-mnist-mlp/mnist_mlp.dc
```

**Примечание:** Пример кода устанавливает `model.device("metal")` на строке 41. Если запускать без поддержки GPU, код автоматически переключится на CPU с предупреждающим сообщением. Для оптимальной производительности скомпилируйте с соответствующим флагом GPU (`--features metal` для macOS или `--features cuda` для Linux/Windows).

## Требования

- Файлы набора данных MNIST должны находиться в `src/lib/ml/datasets/mnist/`:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## API слоёв

Этот пример использует новый API `ml.layer.*` для создания слоёв:

- `ml.layer.linear(in_features, out_features)` - Создаёт линейный (полносвязный) слой
- `ml.layer.relu()` - Создаёт слой активации ReLU
- `ml.layer.softmax()` - Создаёт слой активации Softmax
- `ml.layer.flatten()` - Создаёт слой Flatten (не используется в этом примере)

Слои также могут вызываться напрямую как функции:
```datacode
layer = ml.layer.linear(10, 5)
output = layer(input_tensor)  # Прямой вызов слоя
```

## Поддержка GPU

Этот пример настроен на использование ускорения GPU при наличии. Код включает:
- `model.device("metal")` - Устанавливает устройство на Metal (macOS) или может быть изменено на `"cuda"` (Linux/Windows) или `"cpu"`

**Важно:** Для включения поддержки GPU необходимо скомпилировать проект с соответствующим флагом:
- `--features metal` для macOS
- `--features cuda` для Linux/Windows с GPU NVIDIA

Если поддержка GPU не скомпилирована, код автоматически переключится на выполнение на CPU, но обучение будет значительно медленнее.

## Примечания

- Модель использует разреженную кросс-энтропию (цели - индексы классов 0-9)
- Оптимизатор SGD с коэффициентом обучения 0.001
- Размер пакета 32 для обучения
- Входные изображения нормализованы в диапазон [0, 1] (автоматически выполняется `load_mnist`)
- Слои используют инициализацию He по умолчанию (хорошо для активаций ReLU)

