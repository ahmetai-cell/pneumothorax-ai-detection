import { useState } from 'react'

const TABS = [
  { key: 'original',     label: 'Orijinal' },
  { key: 'segmentation', label: 'Maske Overlay' },
  { key: 'gradcam',      label: 'Grad-CAM' },
]

export default function ImagePanel({ originalImage, segmentationImage, gradcamImage }) {
  const [active, setActive] = useState('segmentation')

  if (!originalImage && !segmentationImage) return null

  const imgMap = {
    original:     originalImage,
    segmentation: segmentationImage,
    gradcam:      gradcamImage,
  }
  const labelMap = {
    original:     'Orijinal X-Ray',
    segmentation: 'Yeşil Overlay — Pnömotoraks Bölgesi',
    gradcam:      'Grad-CAM — Model Dikkat Haritası',
  }

  const current = imgMap[active]

  return (
    <div className="card p-3 space-y-3">
      {/* Tab bar */}
      <div className="flex gap-1 bg-gray-800/60 rounded-lg p-1">
        {TABS.map(({ key, label }) => {
          const available = !!imgMap[key]
          return (
            <button
              key={key}
              onClick={() => available && setActive(key)}
              disabled={!available}
              className={`flex-1 text-xs py-1.5 rounded-md font-medium transition-colors
                ${active === key
                  ? 'bg-gray-700 text-gray-100'
                  : available
                    ? 'text-gray-400 hover:text-gray-200'
                    : 'text-gray-700 cursor-not-allowed'
                }`}
            >
              {label}
            </button>
          )
        })}
      </div>

      {/* Image */}
      <div className="relative">
        <p className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          {labelMap[active]}
        </p>
        {current ? (
          <img
            src={`data:image/png;base64,${current}`}
            alt={labelMap[active]}
            className="w-full rounded-lg object-contain bg-black"
            style={{ maxHeight: 420 }}
          />
        ) : (
          <div className="flex items-center justify-center h-48 rounded-lg bg-gray-800/40 text-gray-600 text-sm">
            Görüntü yok
          </div>
        )}
      </div>

      {/* Hint for segmentation */}
      {active === 'segmentation' && segmentationImage && (
        <p className="text-xs text-gray-500">
          Yeşil alan = model tarafından tespit edilen pnömotoraks bölgesi
        </p>
      )}
      {active === 'gradcam' && gradcamImage && (
        <p className="text-xs text-gray-500">
          Sıcak renkler = modelin kararı en çok etkileyen bölgeler
        </p>
      )}
    </div>
  )
}
