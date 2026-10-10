// Two edits added head to tail above the candidate's row of E, drawn as a
// horizontal axis: each edit's vote is how far it reaches along that axis
// (its shadow). Edit a reaches forward (a positive vote), edit b back (a
// negative one); the stacked bars below show the shadows adding up to the
// sum's shadow, which is the score.
const AXIS_Y = 112, O = 30
const A = { x: 180, y: -72 }   // edit a, from the origin
const B = { x: -70, y: -26 }   // edit b, from a's tip

export function VoteFigure() {
  const tipA = { x: O + A.x, y: AXIS_Y + A.y }
  const tipS = { x: tipA.x + B.x, y: tipA.y + B.y }
  return (
    <svg data-testid="gl-votes-figure" className="gl-votes-figure" viewBox="0 0 320 186" width="320" height="186" role="img"
      aria-label="Two edits added head to tail; each one's shadow on the candidate's row is its vote, and the shadows add up to the score">
      <defs>
        <marker id="gl-vote-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" className="gl-vote-headfill" />
        </marker>
      </defs>
      <line x1={O - 10} y1={AXIS_Y} x2={300} y2={AXIS_Y} className="gl-vote-row" markerEnd="url(#gl-vote-head)" />
      <text x={300} y={AXIS_Y - 6} textAnchor="end" className="gl-vote-label">candidate’s row of E</text>
      <line x1={tipA.x} y1={tipA.y} x2={tipA.x} y2={AXIS_Y + 26} className="gl-vote-drop" />
      <line x1={tipS.x} y1={tipS.y} x2={tipS.x} y2={AXIS_Y + 62} className="gl-vote-drop" />
      <line x1={O} y1={AXIS_Y} x2={tipA.x} y2={tipA.y} className="gl-vote-edit" markerEnd="url(#gl-vote-head)" />
      <line x1={tipA.x} y1={tipA.y} x2={tipS.x} y2={tipS.y} className="gl-vote-edit" markerEnd="url(#gl-vote-head)" />
      <text x={O + A.x * 0.42} y={AXIS_Y + A.y * 0.42 - 8} textAnchor="end" className="gl-vote-label">edit a</text>
      <text x={tipA.x + B.x * 0.5 - 4} y={tipA.y + B.y * 0.5 - 8} textAnchor="end" className="gl-vote-label">edit b</text>
      <line x1={O} y1={AXIS_Y + 20} x2={tipA.x} y2={AXIS_Y + 20} className="gl-vote-share-pos" />
      <text x={tipA.x + 6} y={AXIS_Y + 24} className="gl-vote-label gl-vote-pos">vote of a (+)</text>
      <line x1={tipA.x} y1={AXIS_Y + 38} x2={tipS.x} y2={AXIS_Y + 38} className="gl-vote-share-neg" />
      <text x={tipA.x + 6} y={AXIS_Y + 42} className="gl-vote-label gl-vote-neg">vote of b (−)</text>
      <line x1={O} y1={AXIS_Y + 56} x2={tipS.x} y2={AXIS_Y + 56} className="gl-vote-total" />
      <text x={tipS.x + 6} y={AXIS_Y + 60} className="gl-vote-label">score = sum of the votes</text>
      <circle cx={O} cy={AXIS_Y} r={2.5} className="gl-vote-headfill" />
    </svg>
  )
}
