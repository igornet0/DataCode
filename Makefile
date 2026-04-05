# Makefile для DataCode
# Удобные команды для сборки, тестирования и установки DataCode

.PHONY: help build test run install update uninstall clean dev release examples

OS=$(shell uname -s)
ARCH=$(shell uname -m)

FULL=$(shell cargo pkgid | cut -d\# -f2)

APP_NAME=$(word 1,$(subst @, ,$(FULL)))
VERSION=$(word 2,$(subst @, ,$(FULL)))
# Имя основного бинарника ([[bin]] в Cargo.toml), не путать с именем пакета (APP_NAME)
BIN_NAME=datacode

TARGET_LINUX=x86_64-unknown-linux-gnu
TARGET_WIN=x86_64-pc-windows-gnu
ifeq ($(ARCH),arm64)
TARGET_MAC=aarch64-apple-darwin
else
TARGET_MAC=x86_64-apple-darwin
endif

DIST_DIR=$(shell pwd)/dist
DIST_LINUX_DIR=$(DIST_DIR)/linux
DIST_MACOS_DIR=$(DIST_DIR)/macos
DIST_WINDOWS_DIR=$(DIST_DIR)/windows

# Иконка тома DMG: iconutil из .iconset; если iconset не проходит проверку — готовый .icns
MACOS_VOLICONSET=packaging/macos/datacode-plot.iconset
MACOS_VOLICON_ICNS=packaging/macos/datacode-plot.icns
# App bundle для DMG (иконка в Finder задаётся через CFBundleIconFile + Resources/*.icns)
MACOS_APP_BUNDLE=packaging/macos/DataCode.app
# README для дистрибутивов (установка и доступ к datacode из терминала)
README_DMG=packaging/macos/README-DMG.txt
README_LINUX=packaging/INSTALL-Linux.txt
README_MACOS_TARBALL=packaging/INSTALL-macOS-tarball.txt
README_WINDOWS=packaging/INSTALL-Windows.txt

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

# ------------------------
# BUILD
# ------------------------

build-release:
	@echo "🔨 Сборка DataCode (релизный режим)..."
	@echo "🔨 Сборка DataCode (Linux/Windows)..."
	cross build --release --target $(TARGET_LINUX)
	cross build --release --target $(TARGET_WIN)
	@echo "🔨 Сборка DataCode (macOS)..."
	cargo build --release --target $(TARGET_MAC)

build-release-macos:
	@echo "🔨 Сборка DataCode (релизный режим)..."
	# Linux / Windows через cross (на Apple Silicon — amd64-образ Docker)
	@echo "🔨 Сборка DataCode (Linux/Windows)..."
	DOCKER_DEFAULT_PLATFORM=linux/amd64 cross build --release --target $(TARGET_LINUX)
	DOCKER_DEFAULT_PLATFORM=linux/amd64 cross build --release --target $(TARGET_WIN)
	# macOS (TARGET_MAC зависит от ARCH; при необходимости: TARGET_MAC=x86_64-apple-darwin и rustup target add)
	@echo "🔨 Сборка DataCode (macOS)..."
	cargo build --release --target $(TARGET_MAC)

# Сборка в режиме отладки
build:
	@echo "🔨 Сборка DataCode (режим отладки)..."
	cargo build

# ------------------------
# PREPARE INSTALL SCRIPTS
# ------------------------

prepare-scripts:
	@echo "🔨 Подготовка скриптов установки..."

ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm64)
	@$(MAKE) build-release-macos
else
	@$(MAKE) build-release
endif
else
	@$(MAKE) build-release
endif
	mkdir -p $(DIST_DIR)/tmp

	# Linux/macOS install
	echo '#!/bin/bash' > $(DIST_DIR)/tmp/install.sh
	echo 'set -e' >> $(DIST_DIR)/tmp/install.sh
	echo 'chmod +x $(BIN_NAME)' >> $(DIST_DIR)/tmp/install.sh
	echo 'sudo mv $(BIN_NAME) /usr/local/bin/' >> $(DIST_DIR)/tmp/install.sh
	echo 'echo "$(BIN_NAME) installed"' >> $(DIST_DIR)/tmp/install.sh

	chmod +x $(DIST_DIR)/tmp/install.sh

	# Windows install
	echo '$$InstallDir = "$$env:ProgramFiles\\Datacode"' > $(DIST_DIR)/tmp/install.ps1
	echo 'New-Item -ItemType Directory -Force -Path $$InstallDir' >> $(DIST_DIR)/tmp/install.ps1
	echo 'Copy-Item "$(BIN_NAME).exe" "$$InstallDir\\$(BIN_NAME).exe"' >> $(DIST_DIR)/tmp/install.ps1
	echo '[Environment]::SetEnvironmentVariable("Path", $$env:Path + ";$$InstallDir", "Machine")' >> $(DIST_DIR)/tmp/install.ps1

# ------------------------
# PACKAGE
# ------------------------

package: prepare-scripts
	mkdir -p $(DIST_DIR)

	# Linux
	mkdir -p $(DIST_LINUX_DIR)
	cp target/$(TARGET_LINUX)/release/$(BIN_NAME) $(DIST_DIR)/linux/
	cp $(DIST_DIR)/tmp/install.sh $(DIST_DIR)/linux/
	cp $(README_LINUX) $(DIST_LINUX_DIR)/README.txt

	tar -czf $(DIST_DIR)/$(APP_NAME)-$(VERSION)-linux.tar.gz -C $(DIST_DIR)/linux .

	# macOS
	mkdir -p $(DIST_MACOS_DIR)
	cp target/$(TARGET_MAC)/release/$(BIN_NAME) $(DIST_DIR)/macos/
	cp $(DIST_DIR)/tmp/install.sh $(DIST_DIR)/macos/
	cp $(README_MACOS_TARBALL) $(DIST_MACOS_DIR)/README.txt

	tar -czf $(DIST_DIR)/$(APP_NAME)-$(VERSION)-macos.tar.gz -C $(DIST_DIR)/macos .

	# Windows
	mkdir -p $(DIST_WINDOWS_DIR)
	cp target/$(TARGET_WIN)/release/$(BIN_NAME).exe $(DIST_DIR)/windows/
	cp $(DIST_DIR)/tmp/install.ps1 $(DIST_DIR)/windows/
	cp $(README_WINDOWS) $(DIST_WINDOWS_DIR)/README.txt

	cd $(DIST_DIR)/windows && zip ../$(APP_NAME)-$(VERSION)-windows.zip *

# ------------------------
# OPTIONAL: macOS DMG
# ------------------------

dmg:
	brew install create-dmg || true

	rm -rf $(DIST_DIR)/dmg
	mkdir -p $(DIST_DIR)/dmg
	# Иконка тома — вне папки-источника DMG, иначе в образ попадёт лишний volume.icns
	iconutil -c icns $(MACOS_VOLICONSET) -o $(DIST_DIR)/dmg_volicon.icns || cp $(MACOS_VOLICON_ICNS) $(DIST_DIR)/dmg_volicon.icns
	cp -R $(MACOS_APP_BUNDLE) $(DIST_DIR)/dmg/
	cp target/$(TARGET_MAC)/release/$(BIN_NAME) $(DIST_DIR)/dmg/DataCode.app/Contents/MacOS/datacode
	chmod +x $(DIST_DIR)/dmg/DataCode.app/Contents/MacOS/datacode
	cp $(MACOS_VOLICON_ICNS) $(DIST_DIR)/dmg/DataCode.app/Contents/Resources/datacode-plot.icns
	cp $(README_DMG) $(DIST_DIR)/dmg/README.txt

	create-dmg \
	  --volname "$(APP_NAME)" \
	  --volicon "$(DIST_DIR)/dmg_volicon.icns" \
	  --window-size 660 420 \
	  --icon-size 128 \
	  --icon DataCode.app 140 200 \
	  --hide-extension DataCode.app \
	  --icon README.txt 360 200 \
	  --app-drop-link 520 200 \
	  "$(DIST_DIR)/$(APP_NAME)-$(VERSION).dmg" \
	  "$(DIST_DIR)/dmg"

# ------------------------
# OPTIONAL: WINDOWS INSTALLER (NSIS)
# ------------------------

windows-installer:
	makensis packaging/windows.nsi

# ------------------------
# CLEAN
# ------------------------

clean-dist:
	rm -rf $(DIST_DIR)

# ------------------------
# FULL RELEASE
# ------------------------

release: clean-dist package dmg

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
	@echo "🔨 Пересборка и переустановка DataCode..."
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

print-version:
	@echo "App Name: $(APP_NAME)"
	@echo "Version: $(VERSION)"