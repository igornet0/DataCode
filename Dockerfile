FROM debian:stable-slim

# системные зависимости
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# бинарь языка
COPY datacode /usr/local/bin/datacode

# создаём пользователя (НЕ root)
RUN useradd -m datacode_run
USER datacode_run

WORKDIR /app

# важно: универсальный entrypoint
ENTRYPOINT ["datacode"]