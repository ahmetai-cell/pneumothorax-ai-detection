export default function ResultBadge({ hasPneumothorax, probability }) {
  const pct = (probability * 100).toFixed(1)

  if (hasPneumothorax) {
    return (
      <div className="flex items-center gap-3 px-5 py-4 rounded-xl bg-red-950 border border-red-700">
        <span className="text-2xl">⚠️</span>
        <div>
          <p className="text-red-300 font-bold text-lg leading-tight">
            PNÖMOTORAKS TESPİT EDİLDİ
          </p>
          <p className="text-red-400 text-sm">Güven: %{pct}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 px-5 py-4 rounded-xl bg-green-950 border border-green-700">
      <span className="text-2xl">✅</span>
      <div>
        <p className="text-green-300 font-bold text-lg leading-tight">
          Normal — Pnömotoraks Saptanmadı
        </p>
        <p className="text-green-500 text-sm">Güven: %{pct}</p>
      </div>
    </div>
  )
}
