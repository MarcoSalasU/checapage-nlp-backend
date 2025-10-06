# app.py
# Backend FastAPI para ChecaPage con registro de análisis en Postgres
# Indicadores cubiertos: NPI, TD y FPR (requiere setear ground truth)

import os
import time
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está definido en las variables de entorno")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0.0")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Motor de BD (pool con verificación de conexión)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Esquema SQL: urls y analyses
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS urls (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analyses (
  id                   BIGSERIAL PRIMARY KEY,
  url_id               BIGINT NOT NULL REFERENCES urls(id) ON DELETE CASCADE,
  started_at           TIMESTAMPTZ NOT NULL,
  finished_at          TIMESTAMPTZ NOT NULL,
  duration_ms          INT NOT NULL,
  model_version        TEXT NOT NULL,
  predicted_label      TEXT NOT NULL CHECK (predicted_label IN ('malicious','benign')),
  predicted_confidence NUMERIC(5,4) CHECK (predicted_confidence BETWEEN 0 AND 1),
  ground_truth_label   TEXT NULL CHECK (ground_truth_label IN ('malicious','benign')),
  source               TEXT NULL,
  meta                 JSONB NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyses_url_time ON analyses (url_id, created_at);
"""

def init_db() -> None:
    """Crea tablas si no existen."""
    with engine.begin() as con:
        con.exec_driver_sql(SCHEMA_SQL)

def upsert_url(con, url: str) -> int:
    """Inserta o actualiza una URL y devuelve su id."""
    q = text("""
        INSERT INTO urls (url, last_seen_at)
        VALUES (:url, now())
        ON CONFLICT (url)
        DO UPDATE SET last_seen_at = EXCLUDED.last_seen_at
        RETURNING id;
    """)
    return con.execute(q, {"url": url}).scalar()

def insert_analysis(
    con,
    url_id: int,
    started_at: datetime,
    finished_at: datetime,
    duration_ms: int,
    label: str,
    conf: float,
    source: Optional[str],
    meta: Optional[dict],
) -> int:
    """Inserta una fila de análisis y devuelve su id."""
    q = text("""
        INSERT INTO analyses
        (url_id, started_at, finished_at, duration_ms, model_version,
         predicted_label, predicted_confidence, source, meta)
        VALUES
        (:url_id, :started_at, :finished_at, :duration_ms, :model_version,
         :predicted_label, :predicted_confidence, :source, CAST(:meta AS JSONB))
        RETURNING id;
    """)
    return con.execute(q, {
        "url_id": url_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "model_version": MODEL_VERSION,
        "predicted_label": label,
        "predicted_confidence": round(float(conf), 4),
        "source": source,
        "meta": None if meta is None else json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
    }).scalar()

# ---------------------------------------------------------------------
# Carga de modelo
# ---------------------------------------------------------------------

# Ajusta el nombre del archivo si es diferente en tu repo
model = load_model("phishing_model.keras")

# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------

app = FastAPI(title="ChecaPage Backend", version=MODEL_VERSION)

# CORS: abre todo por ahora; luego restringe a tu extensión/panel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic: esquemas de entrada
class PredictOnly(BaseModel):
    features: list[float] = Field(..., description="Vector de características")

class AnalyzeInput(PredictOnly):
    url: str = Field(..., description="URL analizada")
    source: Optional[str] = Field(default="url_analyzer", description="Origen del análisis")
    meta: Optional[dict] = Field(default=None, description="Datos adicionales opcionales")

# Startup: crear tablas
@app.on_event("startup")
def on_startup() -> None:
    init_db()

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    """Prueba de vida y de conexión a la BD."""
    try:
        with engine.begin() as con:
            con.execute(text("SELECT 1"))
        return {"ok": True, "version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: PredictOnly):
    """
    Solo predice (no guarda en BD).
    Devuelve etiqueta y probabilidad del modelo.
    """
    try:
        features = np.array(data.features).reshape(1, -1)
        prob = model.predict(features, verbose=0)[0][0]
        label = "malicious" if prob > THRESHOLD else "benign"
        return {"resultado": label, "probabilidad": round(float(prob), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze(data: AnalyzeInput):
    """
    Predice, mide tiempo de detección (TD) y registra una fila en 'analyses'.
    Devuelve: analysisId, label, confidence, tdMs y timestamp.
    """
    try:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        features = np.array(data.features).reshape(1, -1)
        prob = model.predict(features, verbose=0)[0][0]
        label = "malicious" if prob > THRESHOLD else "benign"

        finished_at = datetime.now(timezone.utc)
        duration_ms = int((time.perf_counter() - t0) * 1000)

        with engine.begin() as con:
            url_id = upsert_url(con, data.url)
            analysis_id = insert_analysis(
                con, url_id, started_at, finished_at, duration_ms,
                label, prob, data.source, data.meta
            )

        return {
            "analysisId": analysis_id,
            "url": data.url,
            "predictedLabel": label,
            "confidence": round(float(prob), 4),
            "tdMs": duration_ms,
            "modelVersion": MODEL_VERSION,
            "at": finished_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groundtruth")
def set_ground_truth(
    url: str = Body(..., embed=True),
    ground_truth_label: str = Body(..., embed=True)  # 'benign' o 'malicious'
):
    """
    Marca la 'verdad terreno' para calcular FPR.
    Puedes cambiarlo a nivel del último análisis de la URL si lo prefieres.
    """
    if ground_truth_label not in ("benign", "malicious"):
        raise HTTPException(400, "ground_truth_label debe ser 'benign' o 'malicious'")

    # Actualiza todos los análisis de esa URL que no tengan o difieran del GT
    q = text("""
        UPDATE analyses a
        SET ground_truth_label = :gt
        WHERE a.url_id = (SELECT id FROM urls WHERE url=:url)
          AND a.ground_truth_label IS DISTINCT FROM :gt
        RETURNING a.id;
    """)
    with engine.begin() as con:
        rows = con.execute(q, {"url": url, "gt": ground_truth_label}).fetchall()
    return {"updated": len(rows)}

@app.get("/metrics")
def metrics(_from: str, to: str):
    """
    KPIs del periodo:
    - NPI: número de predicciones 'malicious'
    - TD: p50, p95, promedio (ms)
    - FPR: FP / Negativos reales, si hay ground truth
    Parámetros: _from y to en ISO ('2025-10-01' o '2025-10-01T00:00:00Z')
    """
    q = text("""
        WITH base AS (
          SELECT *
          FROM analyses
          WHERE created_at BETWEEN :from AND :to
        ),
        f AS (
          SELECT
            SUM(CASE WHEN predicted_label='malicious' AND ground_truth_label='benign' THEN 1 ELSE 0 END)::float AS fp,
            SUM(CASE WHEN ground_truth_label='benign' THEN 1 ELSE 0 END)::float AS benign_real
          FROM base
          WHERE ground_truth_label IS NOT NULL
        )
        SELECT
          COUNT(*) FILTER (WHERE predicted_label='malicious') AS npi,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) AS td_p50_ms,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS td_p95_ms,
          AVG(duration_ms)::INT AS td_avg_ms,
          (SELECT CASE WHEN benign_real>0 THEN fp/benign_real ELSE NULL END FROM f) AS fpr
        FROM base;
    """)
    with engine.begin() as con:
        row = con.execute(q, {"from": _from, "to": to}).mappings().first()
    return {
        "npi": int(row["npi"] or 0),
        "td_p50_ms": float(row["td_p50_ms"] or 0),
        "td_p95_ms": float(row["td_p95_ms"] or 0),
        "td_avg_ms": int(row["td_avg_ms"] or 0),
        "fpr": None if row["fpr"] is None else float(row["fpr"]),
        "from": _from,
        "to": to
    }
