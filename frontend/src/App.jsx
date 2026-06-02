import { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import AnalysisPage from './pages/AnalysisPage.jsx'
import MetricsPage from './pages/MetricsPage.jsx'

export default function App() {
  const [dark, setDark] = useState(
    () => localStorage.getItem('theme') !== 'light'
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar onToggleTheme={() => setDark(d => !d)} isDark={dark} />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<AnalysisPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
        </Routes>
      </main>
    </div>
  )
}
