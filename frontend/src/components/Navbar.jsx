import { NavLink } from 'react-router-dom'

export default function Navbar() {
  const linkCls = ({ isActive }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
      isActive
        ? 'bg-green-600 text-white'
        : 'text-gray-400 hover:text-white hover:bg-gray-800'
    }`

  return (
    <nav className="sticky top-0 z-50 bg-gray-950 border-b border-gray-800">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2 font-bold text-lg tracking-tight">
          <span className="text-2xl">🫁</span>
          <span className="text-green-400">Pnömotoraks AI</span>
        </div>
        <div className="flex gap-1">
          <NavLink to="/" end className={linkCls}>
            Analiz
          </NavLink>
          <NavLink to="/metrics" className={linkCls}>
            Metrikler
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
