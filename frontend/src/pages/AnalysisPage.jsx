import { useEffect, useState } from 'react'
import DicomUploader from '../components/DicomUploader.jsx'
import ImagePanel from '../components/ImagePanel.jsx'
import ResultBadge from '../components/ResultBadge.jsx'
import TTADetail from '../components/TTADetail.jsx'
import { predict, predictTTA, getMetrics } from '../api/client.js'

// ── Compact model summary banner (global, not per-image) ─────────────────────
function ModelBanner({ metrics }) {
  if (!metrics) return null
  const s = metrics.aggregated ?? {}
  const pick = (...keys) => {
    for (const k of keys) {
      const v = s[k]?.mean ?? s[k]
      if (v != null) return Number(v).toFixed(3)
    }
    return '—'
  }
  return (
    <div className="flex flex-wrap gap-4 px-4 py-2 rounded-lg bg-gray-800/50 border border-gray-700 text-xs text-gray-400">
      <span className="font-semibold text-gray-300 mr-2">Model Performansı:</span>
      <span>Dice <span className="text-green-400 font-mono">{pick('dice','Dice')}</span></span>
      <span>AUC  <span className="text-blue-400  font-mono">{pick('auc','AUC')}</span></span>
      <span>Sensitivite <span className="text-yellow-400 font-mono">{pick('sensitivity','Sensitivity')}</span></span>
      <span>Spesifisite <span className="text-purple-400 font-mono">{pick('specificity','Specificity')}</span></span>
      <span className="ml-auto text-gray-600">EfficientNet-B2 + UNet++ | 5-fold CV</span>
    </div>
  )
}

// ── Risk level ────────────────────────────────────────────────────────────────
function riskLevel(prob) {
  if (prob >= 0.8) return { label: 'YÜKSEK RİSK',  color: 'text-red-400',    bg: 'bg-red-950/40 border-red-700' }
  if (prob >= 0.5) return { label: 'ORTA RİSK',    color: 'text-yellow-400', bg: 'bg-yellow-950/40 border-yellow-700' }
  return               { label: 'DÜŞÜK RİSK',    color: 'text-green-400',  bg: 'bg-green-950/40 border-green-700' }
}

// ── Prediction result card ────────────────────────────────────────────────────
function PredictionResult({ result, mode }) {
  const risk = riskLevel(result.probability)
  return (
    <div className={`card border p-5 space-y-3 ${risk.bg}`}>
      <div className="flex items-start justify-between flex-wrap gap-3">
        {/* YES / NO */}
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-widest mb-1">Tanı</p>
          <ResultBadge hasPneumothorax={result.has_pneumothorax} probability={result.probability} />
        </div>
        {/* Risk + confidence */}
        <div className="text-right">
          <p className={`text-lg font-bold ${risk.color}`}>{risk.label}</p>
          <p className="text-sm text-gray-400 font-mono">
            Güven: {(result.probability * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <p className="text-sm text-gray-300">{result.diagnosis}</p>

      {/* TTA uncertainty */}
      {mode === 'tta' && result.prob_votes && (
        <TTADetail
          probVotes={result.prob_votes}
          probStd={result.prob_std}
          uncertainty={result.uncertainty}
          isUncertain={result.is_uncertain}
        />
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AnalysisPage() {
  const [file,    setFile]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [result,  setResult]  = useState(null)
  const [mode,    setMode]    = useState(null)
  const [metrics, setMetrics] = useState(null)

  // Load global model metrics once (banner only)
  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {})
  }, [])

  async function runAnalysis(useTTA) {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setMode(useTTA ? 'tta' : 'standard')
    try {
      const data = useTTA ? await predictTTA(file) : await predict(file)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-5">

      {/* Page title */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100 mb-1">Akciğer Grafisi Analizi</h1>
        <p className="text-gray-500 text-sm">
          DICOM veya PNG/JPEG yükleyin — yapay zeka pnömotoraks tespiti yapar.
        </p>
      </div>

      {/* Global model performance banner */}
      <ModelBanner metrics={metrics} />

      {/* ── Two-column layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* LEFT: Upload + buttons */}
        <div className="space-y-4">
          <DicomUploader onFile={setFile} file={file} />

          <div className="flex gap-3">
            <button
              className="btn-primary flex-1"
              disabled={!file || loading}
              onClick={() => runAnalysis(false)}
            >
              {loading && mode === 'standard' ? 'Analiz ediliyor…' : 'Standart Analiz'}
            </button>
            <button
              className="btn-secondary flex-1"
              disabled={!file || loading}
              onClick={() => runAnalysis(true)}
            >
              {loading && mode === 'tta' ? 'TTA çalışıyor…' : 'TTA (Daha Güvenilir)'}
            </button>
          </div>

          {/* Spinner */}
          {loading && (
            <div className="flex items-center gap-3 text-gray-400 text-sm">
              <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
              Model çalışıyor, lütfen bekleyin…
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="px-4 py-3 rounded-lg bg-red-950 border border-red-700 text-red-300 text-sm">
              <strong>Hata:</strong> {error}
            </div>
          )}

          {/* Prediction result */}
          {result && <PredictionResult result={result} mode={mode} />}
        </div>

        {/* RIGHT: Image viewer */}
        <div>
          {result ? (
            <ImagePanel
              originalImage={result.original_image}
              segmentationImage={result.segmentation_image}
              gradcamImage={result.gradcam_image}
            />
          ) : (
            <div className="card h-full min-h-[320px] flex items-center justify-center text-gray-600 text-sm">
              Görüntü yüklenince burada görünecek
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
