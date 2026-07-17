# Модуль `ml` (устанавливаемый пакет)

← [Модули](../модули/README.md) · синтаксис импорта: [11-модули-и-импорты](../../0-синтаксис/11-модули-и-импорты.md)

Модуль **`ml`** — **не встроен** в DataCode. Его нужно установить отдельно как пакет DPM из репозитория [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib).

Исходники ML в основном репозитории DataCode зеркалируются в `src/lib/ml/`; публикуемый пакет для пользователей — отдельный git-репозиторий.

---

## Установка

```bash
dpm add ml
```

Пакет описан в реестре DataCode (`datacode_registry_index/config.json`):

- **Имя:** `ml`
- **Источник:** `git+https://github.com/igornet0/ML-Datacode-lib.git`
- **Минимальная версия DataCode:** `>=2.0.0`

После клонирования DPM кладёт пакет в `<окружение>/packages/ml/`. Для нативного `import ml` нужна собранная библиотека:

- macOS: `libml.dylib`
- Linux: `libml.so`
- Windows: `ml.dll`

Сборка (в каталоге пакета `ml`):

```bash
cargo build --release
```

VM ищет `libml.*` в базовом пути скрипта или в `<dpm>/packages/ml/`. Альтернатива — артефакт `.dcmodule` (см. `setup.dcmodule` в корне [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib)).

---

## Импорт и использование

```datacode
import ml

# Тензор
t = ml.tensor([[1, 2], [3, 4]])

# Простая нейросеть
layer1 = ml.layer.linear(784, 128)
layer2 = ml.layer.relu()
layer3 = ml.layer.linear(128, 10)
model = ml.neural_network(ml.sequential([layer1, layer2, layer3]))

# Обучение
loss_history = model.train(x_train, y_train, 10, 32, 0.001, "cross_entropy")
```

Без установленного пакета `import ml` завершится ошибкой: модуль не найден в базовом пути и в пакетах DPM.

---

## Возможности

Пакет предоставляет API для машинного обучения (точный список функций — в документации репозитория [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib)):

| Область | Примеры |
|---------|---------|
| Тензоры | создание, арифметика, матричное умножение |
| Граф вычислений | автоматическое дифференцирование |
| Линейная регрессия | создание и обучение |
| Оптимизаторы | SGD, Adam, … |
| Функции потерь | MSE, Cross Entropy, MAE, … |
| Датасеты | загрузка MNIST и др. |
| Слои и сети | Linear, ReLU, Softmax, Sequential, train/save |

---

## Примеры и документация

| Ресурс | Описание |
|--------|----------|
| [ML-Datacode-lib / examples](https://github.com/igornet0/ML-Datacode-lib/tree/main/examples) | Примеры в репозитории пакета |
| [ML-Datacode-lib / docs](https://github.com/igornet0/ML-Datacode-lib/tree/main/docs) | Документация пакета |
| [1 — Примеры / MNIST](../../1-примеры/11-mnist-mlp.md) | Обзор сценария MNIST MLP (после установки `ml`) |

---

## Отличие от встроенных модулей

| | Встроенные (`plot`, `uuid`, …) | `ml` |
|---|-------------------------------|------|
| Установка | уже в рантайме | `dpm add ml` + сборка native |
| Документация | [2-язык/модули/](../модули/README.md) | этот файл + [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) |
| Глобальные функции | нет (только `import ml`) | нет |

См. также: [загрузка модулей](../../200-разработчикам/module_import_system.md), [`.dcmodule`](../../200-разработчикам/dcmodule-artifact.md).
