# Calibrating the knockout role chips

## Question

`knockoutRoles` (Task 7) turns a knockout sweep into at most five chips with
provisional thresholds: `drop` 0.15, `rise` 0.05, `smallVote` 0.25, `largeVote` 1.0.
Do those values give sensible chips on real sweeps of SmolLM2-135M?

## Setup

Real model (v7, WebGPU, headed Chromium on the screenshots profile), temperature 0,
8 new tokens. Five prompts, two cycles each, one "Measure all 30 layers" per cycle,
then the run exported and read: `sweeps` plus each cycle's `votes.mlp` for the chosen
token. Every sweep took 1.9 s (1.88 to 1.93 s, 31 passes, button click to "done").

## Raw numbers

`flips` is the number of layers whose removal changes the top token. Roles are
counted with the calibrated values below.

| prompt (cycle, chosen token) | baseline | flips | decisive / prep / comp / damp | chips shown |
|---|---|---|---|---|
| chat, France (0, "The") | 0.88 | 4 | 4 / 7 / 0 / 0 | 4 decisive, 1 prep |
| chat, France (5, " Paris") | 0.99 | 2 | 2 / 2 / 2 / 0 | 2 decisive, 2 prep, 1 comp |
| farmer/horse (0, " The") | 0.23 | 17 | 17 / 0 / 0 / 1 | 5 decisive |
| farmer/horse (1, " horse") | 0.51 | 7 | 7 / 3 / 0 / 10 | 5 decisive |
| hungry cat (0, " and") | 0.27 | 12 | 12 / 0 / 1 / 0 | 5 decisive |
| hungry cat (1, " hungry") | 0.21 | 16 | 16 / 0 / 0 / 1 | 5 decisive |
| induction (0, " one") | 0.24 | 22 | 22 / 0 / 0 / 1 | 5 decisive |
| induction (1, " one") | 0.33 | 9 | 9 / 0 / 0 / 2 | 5 decisive |
| free, Japan (0, " Tokyo") | 0.72 | 9 | 9 / 10 / 0 / 3 | 5 decisive |
| free, Japan (1, ".") | 0.51 | 11 | 11 / 0 / 0 / 2 | 5 decisive |

Observations behind the choice:

- The largest direct vote is almost always layer 29 (3 to 30 times the next layer
  except in the Paris cycle, where layers 26 to 28 are all near the maximum). So
  `smallVote` and `largeVote`, being fractions of that maximum, separate "the last
  layer" from "everyone else" and little more.
- Rises (Δp > 0) are common and mostly small: 0.04 to 0.09 in most sweeps, a few real
  ones (induction cycle 1, layer 29, +0.45; farmer cycle 1, layer 28, +0.40).
- Non-flip drops of 0.10 to 0.15 exist (Paris: layers 19, 22, 26) and read as real
  effects on a 0.99 baseline.

## Thresholds

`drop` 0.10, `rise` 0.10, `smallVote` 0.25, `largeVote` 0.5.

- `rise` 0.05 to 0.10: at 0.05 the chat "The" sweep produced three dampening chips for
  rises of 0.06 to 0.09, which are small next to a drop of 0.88. Ten points is the
  smallest rise worth a label.
- `drop` 0.15 to 0.10, matching `rise` so one number ("ten points of probability")
  explains both. It adds the Paris layer 19 chip (−0.12) and nothing else in the top five.
- `smallVote` stays 0.25: moving it to 0.15 or 0.10 changed no top-five list.
- `largeVote` 1.0 to 0.5: at 1.0 only the single largest vote can be "compensated"
  (Paris, layer 28, Δp −0.04). At 0.5 layer 27 (vote 0.97 of the maximum, Δp 0.00) is
  also found, and no other sweep gains a chip in the top five.
- Every sweep shows at least one chip and at most five (the cap). No chip in the top
  five has a change under 0.04.

## Problem the thresholds cannot fix

With a 135M model, removing one MLP often changes the top token, because the top
token's lead is small. Eight of ten sweeps have 9 to 22 flipping layers, and the
cap of five, filled by decisive first, means the chips are then all "decisive": the
other three roles show up only in the two Paris-and-chat sweeps. Layer 0 flips
every sweep (removing it breaks the model, the top token becomes a replacement
character). Two further limits: when a flip is the only criterion, a layer that
barely moves a near-tie counts as decisive; and ties (free prompt, five layers at
−0.72, the model collapses without them) are ordered by layer number.

Possible follow-up, not done here: cap the decisive chips at three so the other
roles have room. On these sweeps that turns, for example, the Japan chips into
three decisive, two preparatory, and adds a dampening chip to most of the others.
