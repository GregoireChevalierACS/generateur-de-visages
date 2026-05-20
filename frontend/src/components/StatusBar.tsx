interface Props {
  messages: { id: number; text: string; type: 'info' | 'error' | 'success' }[]
}

export function StatusBar({ messages }: Props) {
  if (messages.length === 0) return null
  const last = messages[messages.length - 1]
  const colors = {
    info: 'text-zinc-400',
    error: 'text-red-400',
    success: 'text-green-400',
  }
  return (
    <p className={`text-xs text-center ${colors[last.type]}`}>{last.text}</p>
  )
}
