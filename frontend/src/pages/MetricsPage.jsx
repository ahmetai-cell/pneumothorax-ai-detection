import { useEffect, useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from 'recharts'
import MetricCard from '../components/MetricCard.jsx'
import KFoldTable from '../components/KFoldTable.jsx'
import { getResults, getMetrics, getHealth } from '../api/client.js'

// ── Spinner ───────────────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div className="flex items-center gap-3 text-gray-400 text-sm py-8">
      <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
      Yükleniyor…
    </div>
  )
}

// ── Error box ─────────────────────────────────────────────────────────────────
function ErrorBox({ msg }) {
  return (
    <div className="px-4 py-3 rounded-lg bg-red-950 border border-red-700 text-red-300 text-sm">
      <strong>Hata:</strong> {msg}
    </div>
  )
}

// ── Threshold badge ───────────────────────────────────────────────────────────
function ThresholdBadge({ label, value }) {
  return (
    <div className="flex items-center justify-between px-4 py-2 rounded-lg bg-gray-800/60 border border-gray-700">
      <span className="text-xs text-gray-400 uppercase tracking-widest">{label}</span>
      <span className="text-sm font-mono font-bold text-green-400">{value ?? '—'}</span>
    </div>
  )
}

// ── Status dot ────────────────────────────────────────────────────────────────
function StatusDot({ ok }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full mr-2 ${ok ? 'bg-green-400' : 'bg-red-500'}`} />
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MetricsPage() {
  const [results,  setResults]  = useState(null)
  const [metrics,  setMetrics]  = useState(null)
  const [health,   setHealth]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)

  useEffect(() => {
    Promise.allSettled([getResults(), getMetrics(), getHealth()])
      .then(([r, m, h]) => {
        if (r.status === 'fulfilled') setResults(r.value)
        if (m.status === 'fulfilled') setMetrics(m.value)
        if (h.status === 'fulfilled') setHealth(h.value)
        if (r.status === 'rejected' && m.status === 'rejected') {
          setError(r.reason?.message ?? 'Veri yüklenemedi')
        }
      })
      .finally(() => setLoading(false))
  }, [])

  // Summary: prefer /results, fallback to /metrics
  const summary = results?.summary ?? metrics?.aggregated ?? {}

  const pick = (...keys) => {
    for (const k of keys) {
      const v = summary[k]?.mean ?? summary[k]
      if (v != null) return v
    }
    return null
  }
  const pickStd = (...keys) => {
    for (const k of keys) {
      const v = summary[k]?.std
      if (v != null) return v
    }
    return null
  }

  // Radar data
  const radarData = [
    { metric: 'Dice',        value: pick('dice',        'Dice')        ?? 0 },
    { metric: 'IoU',         value: pick('iou',         'IoU')         ?? 0 },
    { metric: 'AUC',         value: pick('auc',         'AUC')         ?? 0 },
    { metric: 'Sensitivite', value: pick('sensitivity', 'Sensitivity') ?? 0 },
    { metric: 'Spesifisite', value: pick('specificity', 'Specificity') ?? 0 },
  ]

  const thresholds = metrics?.thresholds ?? {}
  const device     = metrics?.device ?? health?.device ?? '—'

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-10">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100 mb-1">Model Performans Metrikleri</h1>
        <p className="text-gray-500 text-sm">5-fold çapraz doğrulama — klinik değerlendirme metrikleri</p>
      </div>

      {loading && <Spinner />}
      {error   && <ErrorBox msg={error} />}

      {/* Model status */}
      {health && (
        <div className="card p-4 flex flex-wrap gap-4 items-center justify-between">
          <div className="flex items-center gap-2 text-sm">
            <StatusDot ok={health.ensemble_ready} />
            <span className="text-gray-300">
              {health.ensemble_ready ? 'Ensemble model hazır' : 'Model yüklenmedi'}
            </span>
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-gray-400">
            <span>Cihaz: <span className="text-gray-200 font-mono">{device}</span></span>
            {Object.entries(health.folds ?? {}).map(([fold, ok]) => (
              <span key={fold}>
                <StatusDot ok={ok} />{fold}
              </span>
            ))}
          </div>
        </div>
      )}

      {(Object.keys(summary).length > 0) && (
        <>
          {/* Summary cards — 5 metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <MetricCard
              label="Dice"
              value={pick('dice', 'Dice')}
              std={pickStd('dice', 'Dice')}
              color="green"
            />
            <MetricCard
              label="IoU"
              value={pick('iou', 'IoU')}
              std={pickStd('iou', 'IoU')}
              color="blue"
            />
            <MetricCard
              label="AUC"
              value={pick('auc', 'AUC')}
              std={pickStd('auc', 'AUC')}
              color="purple"
            />
            <MetricCard
              label="Sensitivite"
              value={pick('sensitivity', 'Sensitivity')}
              std={pickStd('sensitivity', 'Sensitivity')}
              color="yellow"
            />
            <MetricCard
              label="Spesifisite"
              value={pick('specificity', 'Specificity')}
              std={pickStd('specificity', 'Specificity')}
              color="green"
            />
          </div>

          {/* Radar + Thresholds */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

            {/* Radar chart */}
            <div className="card p-4">
              <p className="text-sm font-semibold text-gray-300 mb-2">Metrik Radar</p>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 1]}
                    tick={{ fill: '#6b7280', fontSize: 10 }}
                    tickCount={4}
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  <Radar
                    dataKey="value"
                    stroke="#22c55e"
                    fill="#22c55e"
                    fillOpacity={0.25}
                    strokeWidth={2}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                    formatter={(v) => [Number(v).toFixed(4)]}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Thresholds + info */}
            <div className="card p-4 space-y-3">
              <p className="text-sm font-semibold text-gray-300">Eşik Değerleri & Model Bilgisi</p>
              <ThresholdBadge
                label="Sınıflandırma eşiği"
                value={thresholds.cls ?? health?.cls_threshold}
              />
              <ThresholdBadge
                label="Segmentasyon eşiği"
                value={thresholds.seg ?? health?.seg_threshold}
              />
              <ThresholdBadge label="Cihaz"  value={device} />
              {metrics?.aggregated?.loss && (
                <ThresholdBadge
                  label="Val Loss (ort.)"
                  value={typeof metrics.aggregated.loss === 'object'
                    ? metrics.aggregated.loss.mean?.toFixed(4)
                    : Number(metrics.aggregated.loss).toFixed(4)}
                />
              )}
              <p className="text-xs text-gray-600 pt-2">
                Kaynak: {metrics?.aggregated?.source ?? results?.source ?? '—'}
              </p>
            </div>
          </div>
        </>
      )}

      {/* K-Fold table + bar chart */}
      {results?.folds && results.folds.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-200">Fold Detayları</h2>
          <KFoldTable folds={results.folds} />
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && Object.keys(summary).length === 0 && (
        <div className="text-center py-16 space-y-3">
          <p className="text-gray-400 text-lg">Henüz sonuç yok</p>
          <p className="text-gray-600 text-sm">
            Eğitim tamamlandığında metrikler burada görünecek.
          </p>
        </div>
      )}

    </div>
  )
}
