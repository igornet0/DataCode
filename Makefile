# Makefile для DataCode
# Удобные команды для сборки, тестирования и установки DataCode
# ML (libml) не собирается здесь — подключается готовый артефакт через DATACODE_ML_LIB_DIR

.PHONY: help build test run install update uninstall clean dev release examples build-metal build-cuda run-metal run-cuda ml-check ml-link run-with-ml

# Цель по умолчанию
help:
	@echo "🧠 DataCode - Доступные команды"
	@echo "================================"
	@echo ""
	@echo "Разработка:"
	@echo "  make build      - Собрать DataCode в режиме отладки"
	@echo "  make test       - Запустить все тесты"
	@echo "  make run        - Запустить DataCode REPL"
	@echo "  make dev        - Собрать и запустить в режиме разработки"
	@echo ""
	@echo "ML (внешняя сборка, не data-code):"
	@echo "  export DATACODE_ML_LIB_DIR=/path/to/dir   # каталог с libml.dylib / libml.so / ml.dll"
	@echo "  make ml-check   - проверить наличие библиотеки в DATACODE_ML_LIB_DIR"
	@echo "  make ml-link    - скопировать libml в packages/ml (install.sh --ml-artifact-only)"
	@echo "  make run-with-ml FILE=path.dc  - ml-check затем cargo run --release"
	@echo ""
	@echo "Совместимость (alias): build-metal / build-cuda / run-metal / run-cuda —"
	@echo "  только data-code + при необходимости проверка ml; без сборки ML в этом репозитории."
	@echo ""
	@echo "Релиз:"
	@echo "  make release    - Собрать DataCode в релизном режиме"
	@echo "  make install    - Установить DataCode как глобальную команду"
	@echo "  make update     - Обновить зависимости и переустановить data-code"
	@echo "  make uninstall  - Удалить глобальную команду DataCode"
	@echo "  make app-bundle - Создать macOS app bundle с иконкой (только macOS)"
	@echo ""
	@echo "Примеры:"
	@echo "  make examples      - Запустить все файлы примеров"
	@echo "  make run-example   - Запустить конкретный пример (FILE=path/to/file.dc)"
	@echo ""
	@echo "Тестирование:"
	@echo "  make test-cli   - Протестировать командную строку"
	@echo ""
	@echo "Обслуживание:"
	@echo "  make clean      - Очистить артефакты сборки"
	@echo ""
	@echo "Использование после установки:"
	@echo "  datacode                 # Запустить интерактивный REPL"
	@echo "  datacode filename.dc     # Выполнить файл filename.dc"
	@echo "  datacode --help          # Показать справку"
	@echo "  datacode --version       # Показать версию"
	@echo ""
	@echo "Примеры использования:"
	@echo "  datacode hello.dc                                    # Выполнить файл"
	@echo "  datacode examples/01-основы/hello.dc                # Выполнить пример"
	@echo "  datacode examples/01-основы/variables.dc            # Работа с переменными"
	@echo "  datacode examples/02-синтаксис/conditionals.dc     # Условные операторы"
	@echo "  datacode examples/04-функции/simple_functions.dc    # Функции"
	@echo "  datacode examples/05-циклы/for_loops.dc             # Циклы"

# Сборка в режиме отладки
build:
	@echo "🔨 Сборка DataCode (режим отладки)..."
	cargo build

# Сборка в релизном режиме
release:
	@echo "🔨 Сборка DataCode (релизный режим)..."
	cargo build --release

# Сборка только data-code (release). GPU/Metal/CUDA — в отдельно собранном libml, не в data-code.
build-metal build-cuda:
	@echo "🔨 Сборка только data-code (release). ML/libml не собирается в этом репозитории."
	@echo "   Установите DATACODE_ML_LIB_DIR и при необходимости: make ml-link"
	@$(MAKE) release

# Запуск тестов
test:
	@echo "🧪 Запуск тестов..."
	cargo test

# Запуск тестов с тихим выводом
test-quiet:
	@echo "🧪 Запуск тестов (тихий режим)..."
	cargo test --quiet

# Запуск тестов по категориям
test-language:
	@echo "🧪 Запуск тестов языковых возможностей..."
	cargo test language_features

test-data:
	@echo "🧪 Запуск тестов типов данных..."
	cargo test data_types

test-builtins:
	@echo "🧪 Запуск тестов встроенных функций..."
	cargo test builtins

test-errors:
	@echo "🧪 Запуск тестов обработки ошибок..."
	cargo test error_handling

test-performance:
	@echo "🧪 Запуск тестов производительности..."
	cargo test performance

test-integration:
	@echo "🧪 Запуск интеграционных тестов..."
	cargo test integration

# Запуск REPL
run:
	@echo "🚀 Запуск DataCode REPL..."
	cargo run

# Проверка наличия prebuilt libml (DATACODE_ML_LIB_DIR обязателен)
ml-check:
	@if [ -z "$$DATACODE_ML_LIB_DIR" ]; then \
		echo "❌ DATACODE_ML_LIB_DIR не задан."; \
		echo "   Пример: export DATACODE_ML_LIB_DIR=/path/to/dir/with/libml"; \
		exit 1; \
	fi
	@LIB_NAME="$$DATACODE_ML_LIB_NAME"; \
	if [ -z "$$LIB_NAME" ]; then \
		case $$(uname -s) in \
			Darwin) LIB_NAME=libml.dylib ;; \
			MINGW*|MSYS*|CYGWIN*) LIB_NAME=ml.dll ;; \
			*) LIB_NAME=libml.so ;; \
		esac; \
	fi; \
	SRC="$$DATACODE_ML_LIB_DIR/$$LIB_NAME"; \
	if [ ! -f "$$SRC" ]; then \
		echo "❌ Не найдено: $$SRC"; \
		exit 1; \
	fi; \
	echo "✅ ML library: $$SRC"

# Копирование libml в packages/ml (install.sh --ml-artifact-only)
ml-link:
	@chmod +x install.sh
	@./install.sh --ml-artifact-only

# Запуск с проверкой ML-артефакта
run-with-ml:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Укажите файл: make run-with-ml FILE=examples/en/10-mnist-mlp/mnist_mlp.dc"; \
		exit 1; \
	fi
	@$(MAKE) ml-check
	@echo "🚀 Запуск $(FILE)..."
	cargo run --release -- $(FILE)

# Alias: раньше подразумевали сборку libml с Metal/CUDA — теперь только проверка + run
run-metal run-cuda: run-with-ml

# Режим разработки (сборка + запуск)
dev: build run

# Установка как глобальная команда
install:
	@echo "📦 Глобальная установка DataCode..."
	@chmod +x install.sh
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "🍎 macOS detected - will create app bundle after installation"; \
		CREATE_APP_BUNDLE=1 ./install.sh; \
	else \
		./install.sh; \
	fi

# Обновление проекта без полной установки
update:
	@echo "🔄 Обновление DataCode..."
	@echo ""
	@echo "📦 Обновление зависимостей Cargo..."
	@cargo update || (echo "❌ Ошибка: Не удалось обновить зависимости" && exit 1)
	@echo ""
	@echo "🔨 Пересборка и переустановка DataCode (без сборки ML)..."
	@cargo install --path . --force || (echo "❌ Ошибка: Не удалось переустановить DataCode" && exit 1)
	@echo "✅ DataCode обновлен успешно!"
	@if [ "$$(uname)" = "Darwin" ] && [ -d "packaging/macos/DataCode.app" ]; then \
		echo ""; \
		echo "🍎 Обновление macOS app bundle..."; \
		chmod +x packaging/macos/build-app-bundle.sh; \
		./packaging/macos/build-app-bundle.sh || echo "⚠️  Предупреждение: Не удалось обновить app bundle"; \
	fi
	@echo ""
	@echo "🎉 Обновление завершено!"

# Удаление глобальной команды
uninstall:
	@echo "🗑️  Удаление DataCode..."
	@chmod +x uninstall.sh
	@./uninstall.sh

# Запуск файлов примеров
examples:
	@echo "📚 Запуск примеров DataCode..."
	@echo ""
	@echo "🔹 Запуск hello.dc:"
	@cargo run --bin datacode -- examples/01-основы/hello.dc || cargo run -- examples/01-основы/hello.dc
	@echo ""
	@echo "🔹 Запуск variables.dc:"
	@cargo run --bin datacode -- examples/01-основы/variables.dc || cargo run -- examples/01-основы/variables.dc
	@echo ""
	@echo "🔹 Запуск showcase.dc:"
	@cargo run --bin datacode -- examples/06-демонстрации/showcase.dc || cargo run -- examples/06-демонстрации/showcase.dc

# Запуск конкретного примера
run-example:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Укажите файл: make run-example FILE=examples/01-основы/hello.dc"; \
	else \
		echo "🚀 Запуск $(FILE)..."; \
		cargo run --bin datacode -- $(FILE) || cargo run -- $(FILE); \
	fi

# Тестирование командной строки
test-cli: build
	@echo "🧪 Тестирование командной строки..."
	@echo ""
	@echo "🔹 Проверка --help:"
	@./target/debug/datacode --help | head -5
	@echo ""
	@echo "🔹 Проверка --version:"
	@./target/debug/datacode --version
	@echo ""
	@echo "✅ Командная строка работает корректно!"

# Очистка артефактов сборки
clean:
	@echo "🧹 Очистка артефактов сборки..."
	cargo clean

# Проверка форматирования и линтинга кода
check:
	@echo "🔍 Проверка кода..."
	cargo check
	cargo clippy
	cargo fmt --check

# Форматирование кода
format:
	@echo "✨ Форматирование кода..."
	cargo fmt

# Сборка macOS app bundle
app-bundle:
	@echo "🍎 Создание macOS app bundle..."
	@chmod +x packaging/macos/build-app-bundle.sh
	@./packaging/macos/build-app-bundle.sh

# Показать информацию о проекте
info:
	@echo "🧠 Информация о проекте DataCode"
	@echo "==============================="
	@echo "Название: ДатаКод"
	@echo "Версия: $(shell grep '^version' Cargo.toml | cut -d'"' -f2)"
	@echo "Язык: Rust"
	@echo "Лицензия: MIT"
	@echo ""
	@echo "📁 Структура проекта:"
	@echo "  src/           - Исходный код"
	@echo "  examples/      - Примеры .dc файлов"
	@echo "  tests/         - Тестовые файлы"
	@echo ""
	@echo "🔧 Доступные цели: build, test, run, install, examples, app-bundle, ml-check, ml-link"
