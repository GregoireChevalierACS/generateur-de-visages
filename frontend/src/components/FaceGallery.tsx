import { FaceEntry } from '../api'

interface Props {
  faces: FaceEntry[]
  onDelete: (faceId: string) => void
  onClear: () => void
}

export function FaceGallery({ faces, onDelete, onClear }: Props) {
  if (faces.length === 0) return null

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-zinc-300 uppercase tracking-widest">
          Visages mémorisés ({faces.length})
        </h2>
        <button
          onClick={onClear}
          className="text-xs text-zinc-500 hover:text-red-400 transition-colors"
        >
          Tout effacer
        </button>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
        {faces.map(face => (
          <div key={face.face_id} className="group relative aspect-square">
            <img
              src={face.crop_url}
              alt="visage détecté"
              className="w-full h-full object-cover rounded-lg border border-zinc-800"
            />
            <button
              onClick={() => onDelete(face.face_id)}
              className="absolute top-1 right-1 hidden group-hover:flex items-center justify-center w-5 h-5 rounded-full bg-red-600 text-white text-xs leading-none"
              title="Supprimer"
            >
              ×
            </button>
            <span className="absolute bottom-1 left-1 hidden group-hover:block text-[9px] bg-black/70 text-zinc-300 rounded px-1">
              {face.detector}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}
