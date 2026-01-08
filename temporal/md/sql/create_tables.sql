-- Schema de base de datos para datos de criptomonedas
-- PostgreSQL 15+

-- Eliminar tablas si existen (para desarrollo/testing)
DROP TABLE IF EXISTS coin_aggregates CASCADE;
DROP TABLE IF EXISTS coin_data CASCADE;

-- Tabla para almacenar datos diarios de criptomonedas
CREATE TABLE coin_data (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    price_usd NUMERIC(20, 8),
    raw_json JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Asegurar combinación única de moneda y fecha
    CONSTRAINT unique_coin_date UNIQUE (coin_id, date)
);

-- Índices para tabla coin_data
CREATE INDEX idx_coin_data_coin_id ON coin_data(coin_id);
CREATE INDEX idx_coin_data_date ON coin_data(date);
CREATE INDEX idx_coin_data_coin_date ON coin_data(coin_id, date);
CREATE INDEX idx_coin_data_raw_json ON coin_data USING GIN(raw_json);

-- Tabla para almacenar datos agregados mensuales
CREATE TABLE coin_aggregates (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL CHECK (month >= 1 AND month <= 12),
    max_price NUMERIC(20, 8),
    min_price NUMERIC(20, 8),
    avg_price NUMERIC(20, 8),
    num_records INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Asegurar combinación única de moneda, año y mes
    CONSTRAINT unique_coin_year_month UNIQUE (coin_id, year, month)
);

-- Índices para tabla coin_aggregates
CREATE INDEX idx_coin_aggregates_coin_id ON coin_aggregates(coin_id);
CREATE INDEX idx_coin_aggregates_year_month ON coin_aggregates(year, month);
CREATE INDEX idx_coin_aggregates_coin_year_month ON coin_aggregates(coin_id, year, month);

-- Función para actualizar el timestamp de updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para actualizar automáticamente updated_at
CREATE TRIGGER update_coin_data_updated_at
    BEFORE UPDATE ON coin_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_coin_aggregates_updated_at
    BEFORE UPDATE ON coin_aggregates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comentarios para documentación
COMMENT ON TABLE coin_data IS 'Almacena datos diarios de criptomonedas desde la API de CoinGecko';
COMMENT ON COLUMN coin_data.coin_id IS 'Identificador de criptomoneda (ej: bitcoin, ethereum)';
COMMENT ON COLUMN coin_data.date IS 'Fecha del punto de datos';
COMMENT ON COLUMN coin_data.price_usd IS 'Precio en USD';
COMMENT ON COLUMN coin_data.raw_json IS 'Respuesta JSON completa de la API';

COMMENT ON TABLE coin_aggregates IS 'Estadísticas agregadas mensuales de criptomonedas';
COMMENT ON COLUMN coin_aggregates.year IS 'Año de agregación';
COMMENT ON COLUMN coin_aggregates.month IS 'Mes de agregación (1-12)';
COMMENT ON COLUMN coin_aggregates.max_price IS 'Precio máximo en el mes';
COMMENT ON COLUMN coin_aggregates.min_price IS 'Precio mínimo en el mes';
COMMENT ON COLUMN coin_aggregates.avg_price IS 'Precio promedio en el mes';
COMMENT ON COLUMN coin_aggregates.num_records IS 'Número de puntos de datos en el mes';
