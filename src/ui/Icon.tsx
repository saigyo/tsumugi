import type { JSX } from 'react'

export type IconName =
  | 'pin' | 'pinned' | 'link' | 'download' | 'close' | 'import' | 'compare'
  | 'theme-auto' | 'theme-light' | 'theme-dark' | 'info' | 'chevron'
  | 'first' | 'back' | 'play' | 'pause' | 'forward' | 'last' | 'live' | 'prev' | 'next'

const FILL = { fill: 'currentColor', stroke: 'none' } as const
const PIN = 'M6 2.5h4l-.4 4 2.2 2.5H4.2l2.2-2.5z'

// The drawings of the reference sheet (16 x 16, stroked unless a shape carries its own fill).
const SHAPES: Record<IconName, JSX.Element> = {
  close: <path d="M4 4l8 8M12 4l-8 8" />,
  download: <path d="M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10" />,
  import: <path d="M8 11V3M4.5 6.5L8 3l3.5 3.5M3 13.5h10" />,
  link: <path d="M6.5 9.5l3-3M7.2 4.6l1-1a2.6 2.6 0 013.7 3.7l-1 1M8.8 11.4l-1 1a2.6 2.6 0 01-3.7-3.7l1-1" />,
  pin: <path d={`${PIN}M8 9v4.5`} />,
  pinned: <><path {...FILL} d={PIN} /><path d={`${PIN}M8 9v4.5`} /></>,
  compare: <path d="M3 5.5h9.5M10 3l2.5 2.5L10 8M13 10.5H3.5M6 8l-2.5 2.5L6 13" />,
  'theme-light': <><circle cx="8" cy="8" r="2.6" /><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" /></>,
  'theme-dark': <path d="M13 9.6A5.6 5.6 0 016.4 3 5.6 5.6 0 1013 9.6z" />,
  'theme-auto': <><circle cx="8" cy="8" r="5.5" /><path {...FILL} d="M8 2.5a5.5 5.5 0 010 11z" /></>,
  info: <><circle cx="8" cy="8" r="6" /><path d="M8 7.3v3.6" /><circle {...FILL} cx="8" cy="5" r=".9" /></>,
  chevron: <path d="M4.5 6.5L8 10l3.5-3.5" />,
  play: <path {...FILL} d="M5 3.2l7.5 4.8L5 12.8z" />,
  pause: <path {...FILL} d="M4.5 3.5h2.4v9H4.5zM9.1 3.5h2.4v9H9.1z" />,
  back: <path {...FILL} d="M11 3.2L3.5 8l7.5 4.8z" />,
  forward: <><path {...FILL} d="M4 3.2L11.5 8 4 12.8z" /><path d="M12.6 3.5v9" /></>,
  first: <><path d="M2.8 3.5v9" /><path {...FILL} d="M9 3.2L4 8l5 4.8zM14 3.2L9 8l5 4.8z" /></>,
  last: <><path d="M13.2 3.5v9" /><path {...FILL} d="M7 3.2L12 8l-5 4.8zM2 3.2L7 8l-5 4.8z" /></>,
  live: <circle {...FILL} cx="8" cy="8" r="3.2" />,
  prev: <path d="M9.5 4.5L6 8l3.5 3.5" />,
  next: <path d="M6.5 4.5L10 8l-3.5 3.5" />,
}

export function Icon({ name }: { name: IconName }): JSX.Element {
  return (
    <svg className="icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false" data-icon={name}>
      {SHAPES[name]}
    </svg>
  )
}
