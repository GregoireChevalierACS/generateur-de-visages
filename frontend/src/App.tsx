import { useState, useEffect, useCallback } from 'react'
import { api, FaceEntry } from './api'
import { ImageUpload } from './components/ImageUpload'
import { FaceGallery } from './components/FaceGallery'
import { GeneratePanel } from './components/GeneratePanel'
import { StatusBar } from './components/StatusBar'

interface Msg { id: number; text: string; type: 'info' | 'error' | 'success' }

let msgId = 0

export default function App() {
  const [faces, setFaces] = useState<FaceEntry[]>([])
  const [modelReady, setModelReady] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [generatedUrl, setGeneratedUrl] = useState<string | null>(null)
  const [messages, setMessages] = useState<Msg[]>([])

  const push = (text: string, type: Msg['type'] = 'info') =>
    setMessages(prev => [...prev.slice(-4), { id: ++msgId, text, type }])

  const refreshFaces = useCallback(async () => {
    const data = await api.getFaces()
    setFaces(data.faces)
  }, [])

  const refreshStatus = useCallback(async () => {
    const s = await api.generatorStatus()
    setModelReady(s.ready)
  }, [])

  useEffect(() => {
    refreshFaces()
    refreshStatus()
  }, [refreshFaces, refreshStatus])

  const handleFiles = async (files: File[]) => {
    setUploading(true)
    let detected = 0
    for (const file of files) {
      try {
        push(`Upload : ${file.name}…`)
        const uploaded = await api.upload(file)
        push(`Détection dans ${file.name}…`)
        const result = await api.detect(uploaded.id)
        detected += result.face_count
        if (result.face_count === 0) {
          push(`Aucun visage détecté dans ${file.name}`, 'error')
        } else {
          push(`${result.face_count} visage(s) trouvé(s) dans ${file.name}`, 'success')
        }
      } catch (e: unknown) {
        push(`Erreur : ${e instanceof Error ? e.message : String(e)}`, 'error')
      }
    }
    if (detected > 0) await refreshFaces()
    setUploading(false)
  }

  const handleInit = async () => {
    setInitializing(true)
    push('Chargement du modèle SD… (peut prendre 3-5 min)')
    try {
      await api.initGenerator()
      await refreshStatus()
      push('Modèle prêt !', 'success')
    } catch (e: unknown) {
      push(`Erreur chargement modèle : ${e instanceof Error ? e.message : String(e)}`, 'error')
    }
    setInitializing(false)
  }

  const handleGenerate = async (ipScale: number, seed: number | null) => {
    setGenerating(true)
    setGeneratedUrl(null)
    push('Génération en cours…')
    try {
      const result = await api.generate({ ip_scale: ipScale, seed, steps: 30 })
      setGeneratedUrl(result.url)
      push(`Généré à partir de ${result.reference_faces_used} visage(s)`, 'success')
    } catch (e: unknown) {
      push(`Erreur génération : ${e instanceof Error ? e.message : String(e)}`, 'error')
    }
    setGenerating(false)
  }

  const handleDelete = async (faceId: string) => {
    await api.deleteFace(faceId)
    setFaces(prev => prev.filter(f => f.face_id !== faceId))
  }

  const handleClear = async () => {
    await api.clearFaces()
    setFaces([])
  }

  return (
    <div className="min-h-screen px-4 py-10">
      <div className="max-w-2xl mx-auto space-y-8">
        <header className="text-center space-y-1">
          <h1 className="text-2xl font-bold tracking-tight">Générateur de visages</h1>
          <p className="text-sm text-zinc-500">
            Soumets des illustrations · détecte les visages · génère de nouveaux personnages
          </p>
        </header>

        <ImageUpload onFiles={handleFiles} disabled={uploading} />

        <StatusBar messages={messages} />

        <FaceGallery faces={faces} onDelete={handleDelete} onClear={handleClear} />

        <GeneratePanel
          modelReady={modelReady}
          faceCount={faces.length}
          onInit={handleInit}
          onGenerate={handleGenerate}
          generating={generating}
          initializing={initializing}
          generatedUrl={generatedUrl}
        />
      </div>
    </div>
  )
}
