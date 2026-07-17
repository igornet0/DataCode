# Примеры: 11 — MNIST MLP

Машинное обучение с пакетом **`ml`** (устанавливается отдельно, не встроен в DataCode).

## Установка

```bash
dpm add ml
```

Репозиторий: [ML-Datacode-lib](https://github.com/igornet0/ML-Datacode-lib) · документация: [2-язык/модули/ml-модуль](../2-язык/модули/ml-модуль.md)

Примеры `.dc` — в [examples/](https://github.com/igornet0/ML-Datacode-lib/tree/main/examples) репозитория пакета (в основном репозитории DataCode отдельной папки `examples/ru/11-mnist-mlp/` нет).

```bash
# после dpm add ml и сборки libml
datacode path/to/ml/examples/mnist_mlp.dc
```

| Типичные файлы в ML-Datacode-lib | Описание |
|----------------------------------|----------|
| `mnist_mlp.dc` | обучение MLP на MNIST |
| `mnist_model_demo.dc` | демонстрация модели |
