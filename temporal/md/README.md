# Cryptocurrency Data Pipeline

Pipeline modular para extraer, almacenar y analizar datos de criptomonedas desde la API de CoinGecko.

## Descripción

Este proyecto implementa un sistema completo de gestión de datos de criptomonedas que incluye:
- Descarga de datos históricos desde la API de CoinGecko
- Almacenamiento local en archivos JSON
- Persistencia en base de datos PostgreSQL
- Agregaciones mensuales automáticas
- Procesamiento por lotes con soporte concurrente
- Sistema de logging completo

## Requisitos

**Opción A - Docker (Recomendado):**
- Docker y Docker Compose
- API Key de CoinGecko (opcional, modo demo disponible)

**Opción B - Python local:**
- Python 3.10+
- PostgreSQL 15 (o usar Docker solo para PostgreSQL)
- API Key de CoinGecko (opcional)

## Instalación

### 🐳 Opción A: Con Docker (Sin entorno virtual)

**Ventajas:** No necesitas instalar Python ni crear venv, todo está en contenedores.

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd crypto-data-pipeline

# 2. Configurar environment
cp .env.example .env
# Editar .env con tu API key (opcional)

# 3. Iniciar servicios
docker-compose up -d --build

# 4. Verificar
docker ps  # Deberías ver crypto_postgres y crypto_app
```

**Uso:**
```bash
# Ejecutar comandos
docker-compose run --rm app download --date today --coin bitcoin --save-to-db

# Ver guía completa
cat DOCKER.md
```

---

### 🐍 Opción B: Con Python local (Entorno virtual)

**Ventajas:** Más ligero, ejecución más rápida.

#### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd crypto-data-pipeline
```

#### 2. Configurar entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Configurar variables de entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tu configuración
# Nota: COINGECKO_API_KEY es opcional
```

#### 5. Iniciar base de datos PostgreSQL

```bash
# Iniciar contenedor Docker
docker-compose up -d

# Verificar que está corriendo
docker ps
```

El contenedor creará automáticamente las tablas usando el script en `sql/create_tables.sql`.

## Estructura del Proyecto

```
crypto-data-pipeline/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── coingecko_client.py       # Cliente API CoinGecko
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py              # Gestión de conexiones
│   │   ├── models.py                  # Modelos SQLAlchemy
│   │   └── repository.py              # Patrón Repository
│   ├── storage/
│   │   ├── __init__.py
│   │   └── file_handler.py            # Almacenamiento local
│   ├── cli/
│   │   ├── __init__.py
│   │   └── commands.py                # Comandos CLI
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py          # Configuración logging
├── sql/
│   └── create_tables.sql              # Schema de base de datos
├── data/                              # Datos descargados (generado)
├── logs/                              # Logs de aplicación (generado)
├── docker-compose.yml                 # Configuración PostgreSQL
├── requirements.txt                   # Dependencias Python
├── .env.example                       # Ejemplo de configuración
├── .gitignore
└── README.md
```

## Uso

### Comando: download

Descargar datos de una fecha específica para una o más criptomonedas.

```bash
# Descargar Bitcoin para una fecha específica
python -m src.cli.commands download --date 2024-01-15 --coin bitcoin

# Múltiples monedas
python -m src.cli.commands download \
  --date 2024-01-15 \
  --coin bitcoin \
  --coin ethereum \
  --coin cardano

# Usar 'today' para fecha actual
python -m src.cli.commands download --date today --coin bitcoin

# Guardar también en base de datos
python -m src.cli.commands download \
  --date 2024-01-15 \
  --coin bitcoin \
  --save-to-db
```

**Opciones:**
- `--date`: Fecha en formato ISO8601 (YYYY-MM-DD) o 'today'
- `--coin`: ID de la criptomoneda (puede especificarse múltiples veces)
- `--save-to-db`: Guardar en PostgreSQL además del archivo local
- `--data-dir`: Directorio para archivos (default: data)

### Comando: bulk-download

Descarga masiva de datos para un rango de fechas.

```bash
# Descargar rango de fechas
python -m src.cli.commands bulk-download \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --coin bitcoin \
  --coin ethereum

# Con procesamiento concurrente (más rápido)
python -m src.cli.commands bulk-download \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --coin bitcoin \
  --concurrent \
  --max-workers 5

# Guardar en base de datos
python -m src.cli.commands bulk-download \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --coin bitcoin \
  --save-to-db \
  --concurrent
```

**Opciones:**
- `--start-date`: Fecha de inicio (YYYY-MM-DD)
- `--end-date`: Fecha de fin (YYYY-MM-DD)
- `--coin`: ID de criptomoneda (múltiple)
- `--save-to-db`: Guardar en PostgreSQL
- `--concurrent`: Procesamiento concurrente
- `--max-workers`: Número máximo de workers concurrentes (default: 5)

## Configuración CRON

Para ejecutar la descarga automáticamente cada día a las 3am:

### Linux/Mac

```bash
# Editar crontab
crontab -e

# Agregar entrada:
0 3 * * * cd /ruta/al/proyecto && /ruta/al/venv/bin/python -m src.cli.commands download --date today --coin bitcoin --coin ethereum --coin cardano --save-to-db >> /ruta/al/proyecto/logs/cron.log 2>&1
```

### Windows (Task Scheduler)

1. Abrir Task Scheduler
2. Crear tarea básica
3. Trigger: Diariamente a las 3:00 AM
4. Acción: Iniciar programa
   - Programa: `C:\ruta\al\venv\Scripts\python.exe`
   - Argumentos: `-m src.cli.commands download --date today --coin bitcoin --coin ethereum --coin cardano --save-to-db`
   - Directorio: `C:\ruta\al\proyecto`

## Base de Datos

### Conectar a PostgreSQL

```bash
# Desde host
psql -h localhost -p 5432 -U crypto_user -d crypto_data

# Desde contenedor Docker
docker exec -it crypto_postgres psql -U crypto_user -d crypto_data
```

### Tablas

#### coin_data
Almacena datos diarios de criptomonedas.

```sql
SELECT * FROM coin_data WHERE coin_id = 'bitcoin' ORDER BY date DESC LIMIT 10;
```

Columnas:
- `id`: Primary key
- `coin_id`: Identificador de la moneda
- `date`: Fecha del dato
- `price_usd`: Precio en USD
- `raw_json`: Respuesta completa de la API (JSONB)
- `created_at`, `updated_at`: Timestamps

#### coin_aggregates
Agregaciones mensuales automáticas.

```sql
SELECT * FROM coin_aggregates WHERE coin_id = 'bitcoin' ORDER BY year DESC, month DESC;
```

Columnas:
- `id`: Primary key
- `coin_id`: Identificador de la moneda
- `year`, `month`: Período
- `max_price`, `min_price`, `avg_price`: Estadísticas
- `num_records`: Cantidad de registros
- `created_at`, `updated_at`: Timestamps

## Logging

Los logs se almacenan en el directorio `logs/` con el formato:
```
crypto_pipeline_YYYYMMDD_HHMMSS.log
```

Niveles de log configurables via `LOG_LEVEL` en `.env`:
- DEBUG: Información detallada
- INFO: Eventos generales (default)
- WARNING: Advertencias
- ERROR: Errores

## Reproducir Base de Datos

Para obtener una copia exacta de la base de datos con todos los datos históricos:

```bash
# Con Docker (recomendado)
docker-compose up -d
docker exec -i crypto_postgres psql -U crypto_user -d crypto_data < exports/full_backup.sql

# Con PostgreSQL local
psql -U crypto_user -d crypto_data < exports/full_backup.sql
```

Esto carga la estructura completa + todos los datos históricos (agosto-octubre 2025).

## Arquitectura y Buenas Prácticas

Este proyecto implementa:

- **Código Modular**: Separación clara de responsabilidades (API, DB, Storage, CLI)
- **Cohesión**: Cada módulo tiene un propósito específico y bien definido
- **Reutilización**: Componentes independientes y reutilizables
- **Reproducibilidad**: Docker Compose + requirements.txt con versiones fijas
- **Patrón Repository**: Abstracción de acceso a datos
- **Gestión de Conexiones**: Connection pooling y context managers
- **Logging Estructurado**: Sistema centralizado de logs
- **Manejo de Errores**: Try/except con logging apropiado
- **Type Hints**: Anotaciones de tipos para mejor mantenibilidad
- **Documentación**: Docstrings en todas las funciones y clases

## Troubleshooting

### Error de conexión a PostgreSQL

```bash
# Verificar que el contenedor está corriendo
docker ps

# Ver logs del contenedor
docker logs crypto_postgres

# Reiniciar contenedor
docker-compose restart
```

### Error de API Rate Limit

El cliente incluye reintentos automáticos y espera según header `Retry-After`.
Para evitar rate limits:
- Usar API key (aumenta límites)
- Reducir `--max-workers` en modo concurrente
- Agregar delays entre requests

### Archivos no se guardan

Verificar permisos en directorio `data/`:
```bash
chmod 755 data/
```

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `COINGECKO_API_KEY` | API key de CoinGecko | None (modo demo) |
| `COINGECKO_BASE_URL` | URL base de la API | https://api.coingecko.com/api/v3 |
| `POSTGRES_HOST` | Host de PostgreSQL | localhost |
| `POSTGRES_PORT` | Puerto de PostgreSQL | 5432 |
| `POSTGRES_DB` | Nombre de base de datos | crypto_data |
| `POSTGRES_USER` | Usuario de PostgreSQL | crypto_user |
| `POSTGRES_PASSWORD` | Contraseña | crypto_pass |
| `DATA_DIRECTORY` | Directorio de datos | ./data |
| `LOG_LEVEL` | Nivel de logging | INFO |
| `MAX_CONCURRENT_REQUESTS` | Workers concurrentes | 5 |

## Licencia

MIT

## Contacto

Para preguntas o issues, por favor abrir un issue en el repositorio.
