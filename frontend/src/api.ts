const BASE = '/api'

export interface UploadResponse {
  id: string
  filename: string
  url: string
  width: number
  height: number
  size_bytes: number
}

export interface FaceResult {
  face_id: string
  x: number
  y: number
  width: number
  height: number
  confidence: number
  detector: string
  crop_url: string
}

export interface DetectResponse {
  image_id: string
  face_count: number
  faces: FaceResult[]
}

export interface FaceEntry {
  face_id: string
  image_id: string
  crop_url: string
  detector: string
  created_at: string
}

export interface GenerateRequest {
  ip_scale?: number
  steps?: number
  guidance_scale?: number
  seed?: number | null
  prompt_extra?: string
}

export interface GenerateResponse {
  url: string
  seed: number | null
  reference_faces_used: number
  model_ready: boolean
}

export const api = {
  async upload(file: File): Promise<UploadResponse> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
    return res.json()
  },

  async detect(imageId: string): Promise<DetectResponse> {
    const res = await fetch(`${BASE}/detect/${imageId}`, { method: 'POST' })
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
    return res.json()
  },

  async getFaces(): Promise<{ total: number; faces: FaceEntry[] }> {
    const res = await fetch(`${BASE}/faces`)
    if (!res.ok) throw new Error(res.statusText)
    return res.json()
  },

  async deleteFace(faceId: string): Promise<void> {
    await fetch(`${BASE}/faces/${faceId}`, { method: 'DELETE' })
  },

  async clearFaces(): Promise<void> {
    await fetch(`${BASE}/faces`, { method: 'DELETE' })
  },

  async generatorStatus(): Promise<{ ready: boolean; loading: boolean; error: string | null }> {
    const res = await fetch(`${BASE}/generate/status`)
    return res.json()
  },

  async initGenerator(): Promise<{ status: string }> {
    const res = await fetch(`${BASE}/generate/init`, { method: 'POST' })
    return res.json()
  },

  async generate(req: GenerateRequest = {}): Promise<GenerateResponse> {
    const res = await fetch(`${BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
    return res.json()
  },
}
