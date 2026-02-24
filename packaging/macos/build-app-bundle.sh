#!/bin/bash
# Скрипт для создания macOS app bundle с иконкой

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Путь к корню проекта: из packaging/macos -> корень проекта
# Поднимаемся на 2 уровня: macos -> packaging -> корень
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_BUNDLE="$SCRIPT_DIR/DataCode.app"
ICON_SOURCE="$SCRIPT_DIR/datacode-plot.icns"
ICON_DEST="$APP_BUNDLE/Contents/Resources/datacode-plot.icns"

echo "🔨 Создание macOS app bundle для DataCode..."

# Проверка наличия готовой иконки ICNS в проекте
if [ ! -f "$ICON_SOURCE" ]; then
    echo "❌ Иконка не найдена: $ICON_SOURCE"
    echo "   Создайте файл datacode-plot.icns в packaging/macos/ (например, один раз сгенерируйте из PNG через iconutil)."
    exit 1
fi

# Создание структуры директорий
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Сборка приложения в release режиме
echo "📦 Сборка приложения..."
cd "$PROJECT_ROOT"
cargo build --release

# Копирование исполняемого файла
echo "📋 Копирование исполняемого файла..."
cp "$PROJECT_ROOT/target/release/datacode" "$APP_BUNDLE/Contents/MacOS/datacode"
chmod +x "$APP_BUNDLE/Contents/MacOS/datacode"

# Копирование готовой иконки ICNS из проекта
echo "📋 Копирование иконки..."
cp "$ICON_SOURCE" "$ICON_DEST"

echo "✅ App bundle создан: $APP_BUNDLE"
echo "📱 Иконка установлена: $ICON_DEST"
echo ""
echo "Для запуска приложения:"
echo "  open $APP_BUNDLE"
echo ""
echo "Или скопируйте DataCode.app в /Applications"

