-- ===================================================================
-- EJERCICIO 3: Análisis de Datos de Criptomonedas con SQL
-- ===================================================================

-- Query 1: Obtener precio promedio de cada moneda por mes
-- ===================================================================
SELECT
    coin_id,
    DATE_TRUNC('month', date) AS month_start,
    AVG(price_usd) AS avg_price,
    MIN(price_usd) AS min_price,
    MAX(price_usd) AS max_price,
    COUNT(*) AS num_days
FROM coin_data
WHERE price_usd IS NOT NULL
GROUP BY coin_id, DATE_TRUNC('month', date)
ORDER BY coin_id, month_start DESC;


-- Query 2: Recuperación de precio después de caídas consecutivas + Market Cap
-- ===================================================================
-- Calcula cuánto sube el precio después de 3+ días consecutivos de caída
-- e incluye el market cap actual desde el JSON

WITH price_changes AS (
    -- Calcular cambio de precio día a día
    SELECT
        coin_id,
        date,
        price_usd,
        LAG(price_usd, 1) OVER (PARTITION BY coin_id ORDER BY date) AS prev_price_1,
        LAG(price_usd, 2) OVER (PARTITION BY coin_id ORDER BY date) AS prev_price_2,
        LAG(price_usd, 3) OVER (PARTITION BY coin_id ORDER BY date) AS prev_price_3,
        LEAD(price_usd, 1) OVER (PARTITION BY coin_id ORDER BY date) AS next_price,
        raw_json
    FROM coin_data
    WHERE price_usd IS NOT NULL
),
consecutive_drops AS (
    -- Identificar períodos con 3+ días consecutivos de caída
    SELECT
        coin_id,
        date,
        price_usd,
        next_price,
        raw_json,
        CASE
            WHEN price_usd < prev_price_1
             AND prev_price_1 < prev_price_2
             AND prev_price_2 < prev_price_3
            THEN TRUE
            ELSE FALSE
        END AS had_3day_drop
    FROM price_changes
    WHERE prev_price_3 IS NOT NULL
),
recoveries AS (
    -- Calcular recuperación de precio después de caídas
    SELECT
        coin_id,
        date,
        price_usd AS drop_end_price,
        next_price AS recovery_price,
        ((next_price - price_usd) / price_usd * 100) AS price_increase_pct,
        raw_json
    FROM consecutive_drops
    WHERE had_3day_drop = TRUE
      AND next_price IS NOT NULL
      AND next_price > price_usd  -- Solo recuperaciones (no más caídas)
)
SELECT
    coin_id,
    ROUND(AVG(price_increase_pct), 2) AS avg_price_increase_after_3day_drop,
    COUNT(*) AS num_recovery_events,
    -- Extraer market cap del JSON (del último registro)
    (
        SELECT (raw_json->'market_data'->'market_cap'->>'usd')::NUMERIC
        FROM coin_data cd2
        WHERE cd2.coin_id = recoveries.coin_id
        ORDER BY date DESC
        LIMIT 1
    ) AS current_market_cap_usd
FROM recoveries
GROUP BY coin_id
ORDER BY coin_id;
