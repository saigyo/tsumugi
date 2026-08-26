export interface PlayerState {
  cursor: number
  status: 'idle' | 'playing' | 'paused'
  speed: number
  followLive: boolean
  traceLength: number
}

export type PlayerAction =
  | { type: 'play' } | { type: 'pause' }
  | { type: 'stepForward'; auto?: boolean }
  | { type: 'stepBack' }
  | { type: 'seek'; index: number }
  | { type: 'setSpeed'; speed: number }
  | { type: 'traceGrew'; length: number }
  | { type: 'goLive' }
  | { type: 'reset' }

export const initialPlayerState: PlayerState = {
  cursor: -1, status: 'idle', speed: 1, followLive: true, traceLength: 0,
}

const clamp = (i: number, len: number) => (len === 0 ? -1 : Math.max(0, Math.min(i, len - 1)))

export function playerReducer(s: PlayerState, a: PlayerAction): PlayerState {
  switch (a.type) {
    case 'play': return { ...s, status: 'playing' }
    case 'pause': return { ...s, status: 'paused', followLive: false }
    case 'stepForward': {
      const cursor = clamp(s.cursor + 1, s.traceLength)
      return a.auto ? { ...s, cursor } : { ...s, cursor, status: 'paused', followLive: false }
    }
    case 'stepBack': return { ...s, cursor: clamp(s.cursor - 1, s.traceLength), status: 'paused', followLive: false }
    case 'seek': return { ...s, cursor: clamp(a.index, s.traceLength), followLive: false }
    case 'setSpeed': return { ...s, speed: a.speed }
    case 'traceGrew': return { ...s, traceLength: a.length }
    case 'goLive': return { ...s, cursor: s.traceLength - 1, followLive: true, status: 'playing' }
    case 'reset': return { ...initialPlayerState, speed: s.speed, status: 'playing' }
  }
}
