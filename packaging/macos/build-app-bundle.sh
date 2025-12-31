#!/bin/bash
# Скрипт для создания macOS app bundle с иконкой

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Путь к корню проекта: из packaging/macos -> корень проекта
# Поднимаемся на 2 уровня: macos -> packaging -> корень
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_BUNDLE="$SCRIPT_DIR/DataCode.app"
ICON_SOURCE="$PROJECT_ROOT/src/lib/plot/icon/datacode-plot.png"
ICON_DEST="$APP_BUNDLE/Contents/Resources/datacode-plot.icns"
ICONSET_DIR="$SCRIPT_DIR/datacode-plot.iconset"

echo "🔨 Создание macOS app bundle для DataCode..."

# Проверка наличия исходной иконки
if [ ! -f "$ICON_SOURCE" ]; then
    echo "❌ Иконка не найдена: $ICON_SOURCE"
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

# Создание iconset для конвертации в ICNS
echo "🎨 Создание iconset из PNG..."
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

# Создание различных размеров иконок для iconset
sips -z 16 16     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16.png"
sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16@2x.png"
sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32.png"
sips -z 64 64     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32@2x.png"
sips -z 128 128   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128.png"
sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128@2x.png"
sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256.png"
sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256@2x.png"
sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512.png"
sips -z 1024 1024 "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512@2x.png"

# Конвертация iconset в ICNS
echo "🔄 Конвертация iconset в ICNS..."
iconutil -c icns "$ICONSET_DIR" -o "$ICON_DEST"

# Удаление временного iconset
rm -rf "$ICONSET_DIR"

echo "✅ App bundle создан: $APP_BUNDLE"
echo "📱 Иконка установлена: $ICON_DEST"
echo ""
echo "Для запуска приложения:"
echo "  open $APP_BUNDLE"
echo ""
echo "Или скопируйте DataCode.app в /Applications"

