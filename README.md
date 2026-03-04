# tourism-dwh

Хранилище данных и Streamlit-приложение для анализа и прогнозирования туристических потоков.

## 1. Быстрый запуск (Docker)

Требования:
- Docker Desktop
- Docker Compose

Запуск:
```bash
docker compose up --build -d
```

После запуска:
- Приложение: `http://localhost:8501`
- PostgreSQL: `localhost:5433`

Остановка:
```bash
docker compose down
```

Полный сброс БД:
```bash
docker compose down -v
docker compose up --build -d
```

## 2. Локальный запуск (без Docker для app)

Требования:
- Python 3.10
- PostgreSQL (локально или в Docker)

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run .\app\app.py
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run ./app/app.py
```

## 3. Переменные окружения

Используются:
- `PG_HOST`
- `PG_PORT`
- `PG_DB`
- `PG_USER`
- `PG_PASSWORD`

В `docker-compose.yml` есть значения по умолчанию.

## 4. ML-команды

Переобучение NN:
```bash
python -m src.ml.retrain_nn_gpu --device auto --epochs 500 --lr 0.0025 --patience 40 --model-version vYYYYMMDD_exp1 --notes "experiment"
```

Запуск NN-прогноза:
```bash
python -m src.ml.forecast_yearly_nn --horizon 5 --retrain --device auto
```

## 5. Частые ошибки и решения

### `streamlit` не распознано
```bash
python -m streamlit run app/app.py
```

### `ModuleNotFoundError` (например `plotly`, `fpdf`, `bcrypt`)
```bash
python -m pip install -r requirements.txt
```
или:
```bash
python -m pip install plotly fpdf2 bcrypt
```

### Ошибка подключения к БД (`localhost:5433`)
Проверьте контейнер:
```bash
docker compose ps
```
Перезапуск:
```bash
docker compose down -v
docker compose up -d postgres
```
Дождитесь статуса `healthy`.

### CUDA ошибка в PyTorch
Если `torch.cuda.is_available() is False`:
```bash
python -m src.ml.retrain_nn_gpu --device auto
```

Если `no kernel image is available for execution on the device`:
- установите сборку PyTorch/CUDA, совместимую с вашей видеокартой.

