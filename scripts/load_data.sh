#!/bin/bash

set -e

echo "Загрузка данных в базу данных..."

if ! docker compose ps | grep -q "tourism_app.*Up"; then
    echo "Ошибка: контейнер tourism_app не запущен"
    exit 1
fi

echo "1. Загрузка dim_date..."
docker compose exec -T app python -m src.etl.seed_dim_date

echo "2. Подготовка данных World Bank..."
docker compose exec -T app python -m src.etl.build_world_bank_tourism

echo "3. Загрузка dim_destination..."
docker compose exec -T app python -m src.etl.load_dim_destination

echo "4. Обновление country_code..."
docker compose exec -T app python -m src.etl.update_dim_destination_country_code

echo "5. Загрузка fact_flow..."
docker compose exec -T app python -m src.etl.load_fact_flow

echo "Готово! Данные загружены."

