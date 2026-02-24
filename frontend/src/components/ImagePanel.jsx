export default function ImagePanel({ originalImage, segmentationImage }) {
  if (!originalImage && !segmentationImage) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
      <div className="card p-3">
        <p className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Orijinal Görüntü
        </p>
        {originalImage ? (
          <img
            src={`data:image/png;base64,${originalImage}`}
            alt="Orijinal X-ray"
            className="w-full rounded-lg object-contain bg-black"
            style={{ maxHeight: 400 }}
          />
        ) : (
          <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
            Görüntü yok
          </div>
        )}
      </div>

      <div className="card p-3">
        <p className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wide">
          Yeşil Overlay (Pnömotoraks Bölgesi)
        </p>
        {segmentationImage ? (
          <img
            src={`data:image/png;base64,${segmentationImage}`}
            alt="Segmentasyon overlay"
            className="w-full rounded-lg object-contain bg-black"
            style={{ maxHeight: 400 }}
          />
        ) : (
          <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
            Görüntü yok
          </div>
        )}
      </div>
    </div>
  )
}
