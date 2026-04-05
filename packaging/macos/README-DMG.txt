DataCode — установка с образа DMG
====================================

1) Установка приложения
   • Перетащите «DataCode.app» в папку «Программы» (Applications).

2) Команда datacode в терминале (чтобы запускать скрипты: datacode файл.dc)
   После копирования .app команда datacode в PATH не появляется автоматически.
   Выберите один способ:

   Способ A — симлинк (удобно для большинства пользователей):
     sudo ln -sf /Applications/DataCode.app/Contents/MacOS/datacode /usr/local/bin/datacode

   Способ B — добавить каталог с бинарником в PATH (zsh по умолчанию в macOS):
     echo 'export PATH="/Applications/DataCode.app/Contents/MacOS:$PATH"' >> ~/.zshrc
     source ~/.zshrc
     (для bash используйте ~/.bash_profile вместо ~/.zshrc)

3) Проверка
     datacode --version
     datacode путь/к/скрипту.dc

Если /usr/local/bin недоступен, убедитесь, что каталог существует:
  sudo mkdir -p /usr/local/bin
