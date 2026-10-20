// Shared frame constants for the step-by-step charts (#17), kept apart from
// MlpStepCharts.tsx so that file exports only components.

export const CHART_W = 720
export const X0 = 36
export const X1 = 710
export interface Hover { neuron: number; chart: 'scatter' | 'silu' | 'product'; x: number; y: number; height: number }

export const gateX = ([lo, hi]: [number, number]) => (v: number) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0)
