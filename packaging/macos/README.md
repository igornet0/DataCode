# macOS App Bundle для DataCode

Этот каталог содержит скрипты и файлы для создания macOS app bundle с иконкой приложения.

## Структура

```
macos/
├── DataCode.app/          # App bundle (создается автоматически)
│   └── Contents/
│       ├── Info.plist     # Конфигурация приложения
│       ├── MacOS/         # Исполняемый файл
│       └── Resources/     # Иконка приложения (.icns)
├── build-app-bundle.sh    # Скрипт сборки app bundle
└── README.md              # Этот файл
```

## Использование

### Сборка app bundle

```bash
# Через Makefile
make app-bundle

# Или напрямую
./packaging/macos/build-app-bundle.sh
```

Скрипт автоматически:
1. Соберет приложение в release режиме
2. Создаст структуру app bundle
3. Скопирует исполняемый файл
4. Конвертирует PNG иконку в ICNS формат
5. Установит иконку в app bundle

### Запуск приложения

После сборки вы можете запустить приложение:

```bash
# Открыть через Finder
open packaging/macos/DataCode.app

# Или скопировать в Applications
cp -r packaging/macos/DataCode.app /Applications/
```

### Иконка

Иконка берется из `src/lib/plot/icon/datacode-plot.png` и автоматически конвертируется в формат ICNS с различными размерами для macOS.

## Требования

- macOS (для использования `sips` и `iconutil`)
- Rust и Cargo (для сборки приложения)
- Исходная иконка должна существовать: `src/lib/plot/icon/datacode-plot.png`

## Примечания

- App bundle создается в `packaging/macos/DataCode.app`
- При `make install` на macOS app bundle создается автоматически
- После копирования в `/Applications` иконка появится в Dock при запуске
- Если иконка не обновляется, попробуйте очистить кэш:
  ```bash
  sudo rm -rf /Library/Caches/com.apple.iconservices.store
  killall Dock
  ```

