# Makefile для DataCode
# Удобные команды для сборки, тестирования и установки DataCode

.PHONY: help build test run install update uninstall clean dev release examples build-metal build-cuda run-metal run-cuda

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
	@echo "GPU поддержка:"
	@echo "  make build-metal - Собрать с поддержкой Metal (macOS)"
	@echo "  make build-cuda  - Собрать с поддержкой CUDA (Linux/Windows)"
	@echo "  make run-metal   - Запустить с Metal (FILE=path/to/file.dc)"
	@echo "  make run-cuda    - Запустить с CUDA (FILE=path/to/file.dc)"
	@echo ""
	@echo "Релиз:"
	@echo "  make release    - Собрать DataCode в релизном режиме"
	@echo "  make install    - Установить DataCode как глобальную команду"
	@echo "  make update     - Обновить DataCode без полной установки (зависимости + пересборка + переустановка)"
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

# Сборка с поддержкой Metal (macOS)
build-metal:
	@echo "🔨 Сборка DataCode с поддержкой Metal (macOS)..."
	cargo build --features metal

# Сборка с поддержкой CUDA (Linux/Windows)
build-cuda:
	@echo "🔨 Сборка DataCode с поддержкой CUDA (Linux/Windows)..."
	cargo build --features cuda

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

# Запуск с поддержкой Metal (macOS)
run-metal:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Укажите файл: make run-metal FILE=examples/en/10-mnist-mlp/mnist_mlp.dc"; \
	else \
		echo "🚀 Запуск $(FILE) с Metal GPU..."; \
		cargo run --features metal -- $(FILE); \
	fi

# Запуск с поддержкой CUDA (Linux/Windows)
run-cuda:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Укажите файл: make run-cuda FILE=examples/en/10-mnist-mlp/mnist_mlp.dc"; \
	else \
		echo "🚀 Запуск $(FILE) с CUDA GPU..."; \
		cargo run --features cuda -- $(FILE); \
	fi

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
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "🍎 macOS detected - updating with Metal GPU support"; \
		echo "🔨 Пересборка и переустановка DataCode с Metal..."; \
		cargo install --path . --features metal --force || (echo "❌ Ошибка: Не удалось переустановить DataCode" && exit 1); \
		echo "✅ DataCode обновлен успешно!"; \
		echo ""; \
		if [ -d "packaging/macos/DataCode.app" ]; then \
			echo "🍎 Обновление macOS app bundle..."; \
			chmod +x packaging/macos/build-app-bundle.sh; \
			./packaging/macos/build-app-bundle.sh || echo "⚠️  Предупреждение: Не удалось обновить app bundle"; \
		fi; \
	elif [ "$$(uname)" = "Linux" ]; then \
		echo "🐧 Linux detected - updating with CUDA GPU support"; \
		echo "🔨 Пересборка и переустановка DataCode с CUDA..."; \
		cargo install --path . --features cuda --force || (echo "❌ Ошибка: Не удалось переустановить DataCode" && exit 1); \
		echo "✅ DataCode обновлен успешно!"; \
	else \
		echo "🔨 Пересборка и переустановка DataCode..."; \
		cargo install --path . --force || (echo "❌ Ошибка: Не удалось переустановить DataCode" && exit 1); \
		echo "✅ DataCode обновлен успешно!"; \
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
	@echo "🔧 Доступные цели: build, test, run, install, examples, app-bundle"
