// Some vocabulary entries are part of a UTF-8 character (byte-level BPE splits
// rare characters into bytes), so decode([id]) gives "�" and every such token
// looks alike. The raw vocabulary piece still has the bytes, written in GPT-2's
// byte alphabet: this module maps it back and spells out the stray bytes as
// <0xE2>. The pieces come from the tokenizer the app loads at startup; until
// then texts are shown as decoded.

// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the other
// 68 are shifted to U+0100 onwards in byte order.
const BYTE_OF = new Map<string, number>()
{
  let shifted = 0
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff)
    BYTE_OF.set(String.fromCodePoint(printable ? b : 256 + shifted++), b)
  }
}

export function pieceBytes(piece: string): number[] | null {
  const bytes: number[] = []
  for (const ch of piece) {
    const b = BYTE_OF.get(ch)
    if (b === undefined) return null
    bytes.push(b)
  }
  return bytes
}

const hex = (b: number) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`

const strict = new TextDecoder('utf-8', { fatal: true })
const decodeStrict = (seq: number[]) => { try { return strict.decode(new Uint8Array(seq)) } catch { return null } }

// Whole, valid UTF-8 characters are decoded; every other byte becomes <0xNN>.
export function readableBytes(bytes: number[]): string {
  let out = ''
  for (let i = 0; i < bytes.length;) {
    const b = bytes[i]
    const len = b < 0x80 ? 1 : b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3 : b >= 0xf0 && b <= 0xf4 ? 4 : 0
    const seq = bytes.slice(i, i + len)
    const whole = len > 0 && seq.length === len && seq.slice(1).every((c) => (c & 0xc0) === 0x80)
    const text = whole ? decodeStrict(seq) : null
    if (text !== null) { out += text; i += len } else { out += hex(b); i += 1 }
  }
  return out
}

let pieceOf: ((id: number) => string | undefined) | null = null

export function setTokenPieces(lookup: ((id: number) => string | undefined) | null): void { pieceOf = lookup }

export function readableTokenText(id: number, text: string): string {
  if (!text.includes('�') || !pieceOf) return text
  const piece = pieceOf(id)
  const bytes = piece === undefined ? null : pieceBytes(piece)
  return bytes ? readableBytes(bytes) : text
}
