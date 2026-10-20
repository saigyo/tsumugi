import { Formula } from '../../viz/Formula'
import { SiluPlot } from '../figures/SiluPlot'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const mlp: GlossaryEntry = {
  id: 'mlp',
  title: 'MLP',
  summary: 'The per-token step in each layer: widen the vector, apply a nonlinearity, narrow it back.',
  stage: 'layers',
  related: ['layer', 'edit', 'attention', 'votes', 'mlp-neuron'],
  body: () => (
    <>
      <p>MLP stands for multi-layer perceptron, also called the feed-forward network. It works on each token’s
        vector on its own, with no mixing between positions. Its result is added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <p>For one token, in five steps:</p>
      <div data-testid="gl-mlp-steps">
        <Formula tex={String.raw`\begin{aligned}
          n &= \operatorname{norm}(x) && \in \mathbb{R}^{576} \\
          g &= W_{\text{gate}} \cdot n && \in \mathbb{R}^{1536} \\
          u &= W_{\text{up}} \cdot n && \in \mathbb{R}^{1536} \\
          h &= \operatorname{SiLU}(g) \odot u && \in \mathbb{R}^{1536} \\
          \text{edit} &= W_{\text{down}} \cdot h && \in \mathbb{R}^{576}
        \end{aligned}`} />
      </div>
      <Section id="symbols" title="What the symbols mean">
        <dl className="gl-symbols">
          <dt><Formula inline tex="x" /></dt>
          <dd>The token’s vector as it arrives from the residual stream: 576 numbers.</dd>
          <dt><Formula inline tex={String.raw`\operatorname{norm}`} /></dt>
          <dd>RMSNorm: divides the vector by its root-mean-square size, then multiplies each of the 576 numbers by
            a learned scale. The result <Formula inline tex="n" /> has a standard size, however large <Formula inline tex="x" /> has grown.</dd>
          <dt><Formula inline tex={String.raw`W_{\text{gate}}`} /></dt>
          <dd>A learned weight matrix of 1536 rows × 576 columns. The subscript is only its name.{' '}
            <Formula inline tex={String.raw`W_{\text{gate}} \cdot n`} /> is a matrix–vector product: each of the 1536 results is
            the dot product of one row of the matrix with <Formula inline tex="n" />, so each row acts as a pattern detector
            that scores how strongly <Formula inline tex="n" /> matches it.</dd>
          <dt><Formula inline tex={String.raw`W_{\text{up}}`} /></dt>
          <dd>A second matrix of the same shape, with its own learned rows. It produces the 1536 values{' '}
            <Formula inline tex="u" /> that may be passed on.</dd>
          <dt><Formula inline tex={String.raw`\operatorname{SiLU}`} /></dt>
          <dd>A function applied to each of the 1536 gate scores separately:{' '}
            <Formula inline tex={String.raw`\operatorname{SiLU}(z) = \dfrac{z}{1 + e^{-z}}`} />. Negative scores come out near
            zero; positive scores come out roughly unchanged.
            <SiluPlot /></dd>
          <dt><Formula inline tex={String.raw`\odot`} /></dt>
          <dd>Element-wise product: position 1 times position 1, position 2 times position 2, and so on. Where the
            gate is near zero, the matching value of <Formula inline tex="u" /> is blocked; where it is large, the value passes, scaled by the gate. SiLU has no upper limit, so that factor can be
            larger than 1.</dd>
          <dt><Formula inline tex={String.raw`W_{\text{down}}`} /></dt>
          <dd>A learned matrix of 576 rows × 1536 columns that turns the 1536 values back into a 576-number
            edit.</dd>
        </dl>
      </Section>
      <p>The 1536 positions of <Formula inline tex="h" /> are often called neurons. This gated design is called SwiGLU; there are
        no bias terms. A common reading: the rows of the gate and up matrices detect patterns in the token’s
        vector, and each column of <Formula inline tex={String.raw`W_{\text{down}}`} /> is the information written back when its
        neuron is active.</p>
      <p>The three matrices hold 3 × 576 × 1536 ≈ 2.7 million weights per layer, about 60% of the whole model
        across its 30 layers.</p>
      <Section id="silu-not-switch" title="Why SiLU and not a switch">
        <p>A hard switch (ReLU) would pass every positive gate score and set every negative one to exactly zero. SiLU is
          smooth instead: negative scores give small negative values, down to about −0.28 at a score of −1.28, and large
          positive scores pass almost unchanged. Because the curve has no flat zero region, training can still adjust
          neurons whose gate is currently negative, and their small outputs still add a little to the edit.</p>
      </Section>
      <Section id="two-projections" title="Why two projections">
        <p>The gate and up matrices read the same normalized vector but learn different rows, so a neuron’s two values
          are mostly unrelated. The gate decides how much passes, the up value what passes and with which sign. In the
          last layers the largest activations come from gates around 10, where SiLU barely changes the value: there the
          gate sets the size and the up value mostly the sign.</p>
      </Section>
      <Section id="early-layers" title="Early layers look different">
        <p>In early layers most gate scores are negative and the activation is spread over many neurons: on a typical
          prompt, half of layer 5’s squared activation needs about a hundred neurons, while layer 29 needs about ten.
          The last layers concentrate on a few neurons with large gates, which is also where the per-layer votes grow.</p>
      </Section>
      <p>In a real run the Layers detail shows what the MLPs do: <em>How much each layer changes the vector</em>{' '}
        compares each MLP edit with the stream, <em>What each layer voted for</em> shows how much each layer’s
        MLP added to each candidate’s score (see <Term id="votes">per-layer votes</Term>), and{' '}
        <em>Inside one MLP</em> splits one layer’s vote among its <Term id="mlp-neuron">neurons</Term>, and{' '}
        <em>The MLP, step by step</em> shows one layer’s five steps with the model’s own numbers.</p>
    </>
  ),
}
