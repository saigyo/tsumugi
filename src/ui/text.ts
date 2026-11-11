/** The text with its first letter in upper case: a name used inside a sentence, shown on its own as a label. */
export const capitalised = (text: string): string => text.charAt(0).toUpperCase() + text.slice(1)
