import { useState } from 'react'
import DicomUploader from '../components/DicomUploader.jsx'
import ImagePanel from '../components/ImagePanel.jsx'
import ResultBadge from '../components/ResultBadge.jsx'
import MetricCard from '../components/MetricCard.jsx'
import TTADetail from '../components/TTADetail.jsx'
import { predict, predictTTA } from '../api/client.js'

export default function AnalysisPage() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [mode, setMode] = useState(null) // 'standard' | 'tta'

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
      const detail = err.response?.data?.detail ?? err.message
      setError(String(detail))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100 mb-1">Akciğer Grafisi Analizi</h1>
        <p className="text-gray-500 text-sm">
          DICOM veya PNG/JPEG dosyası yükleyin, yapay zeka pnömotoraks tespiti yapar.
        </p>
      </div>

      <DicomUploader onFile={setFile} file={file} />

      <div className="flex gap-3">
        <button
          className="btn-primary"
          disabled={!file || loading}
          onClick={() => runAnalysis(false)}
        >
          {loading && mode === 'standard' ? 'Analiz ediliyor…' : 'Standart Analiz'}
        </button>
        <button
          className="btn-secondary"
          disabled={!file || loading}
          onClick={() => runAnalysis(true)}
        >
          {loading && mode === 'tta' ? 'TTA çalışıyor…' : 'TTA (Daha Güvenilir)'}
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-gray-400 text-sm">
          <div className="w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin" />
          Model çalışıyor, lütfen bekleyin…
        </div>
      )}

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-950 border border-red-700 text-red-300 text-sm">
          <strong>Hata:</strong> {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <hr className="border-gray-800" />

          <ResultBadge
            hasPneumothorax={result.has_pneumothorax}
            probability={result.probability}
          />

          {/* Metrik kartları */}
          {result.dice != null || result.auc != null ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard label="Dice"        value={result.dice}        color="green"  />
              <MetricCard label="AUC"         value={result.auc}         color="blue"   />
              <MetricCard label="Sensitivite" value={result.sensitivity} color="yellow" />
              <MetricCard label="Spesifisite" value={result.specificity} color="purple" />
            </div>
          ) : null}

          <ImagePanel
            originalImage={result.original_image}
            segmentationImage={result.segmentation_image}
          />

          {/* TTA detayları */}
          {mode === 'tta' && result.prob_votes && (
            <TTADetail
              probVotes={result.prob_votes}
              probStd={result.prob_std}
              uncertainty={result.uncertainty}
              isUncertain={result.is_uncertain}
            />
          )}
        </div>
      )}
    </div>
  )
}
