# Semistructural Project Handoff Notes

This note packages the content and working style from the July 2026 discussion so it can be reused in future conversations. It is written as a handoff: use it as context for the project, and as a style guide for future explanations.

## Working Style To Continue

Use the following format for technical writeups:

1. Main finding first.
2. Then a technical but jargon-light explanation.
3. Keep the notation close to the paper being discussed.
4. End by tying the point back to the semistructural project.
5. When useful, provide a LaTeX-ready block.

Preferred tone:

- Direct and explanatory.
- Avoid overclaiming.
- Keep the economic object distinct from the statistical object.
- When a notation mismatch appears, correct it explicitly and use the paper's notation.

Reusable prompt for future chats:

```text
Please explain this in the same style as the semistructural AKM project notes:
main finding first, then a technical but jargon-free explanation, using the
paper's own notation, and ending with how the point connects to the project.
If useful, give a LaTeX-ready version.
```

## Core Project Definition Of Semistructural

The Wolf macro slides sharpen the definition:

```latex
\text{semistructural}
=
\text{empirical/statistical interface}
+
\text{limited economic structure for interpretation}
-
\text{full structural model}.
```

In macro, the empirical interface is:

```latex
\text{IRFs, shocks, autocovariances, policy causal effects}.
```

The structure is just enough to say when those empirical objects answer policy counterfactuals. The researcher does not estimate a full DSGE model.

For this labor project, the analogous interface is:

```latex
\text{AKM/edge effects, cycle restrictions, match graph, variance components}.
```

The structural content is just enough to interpret deviations from AKM as match effects, sorting, or comparative advantage, without estimating a full search/matching model.

One-sentence version:

```latex
The semistructural approach estimates a statistical interface that many economic
models map into, then uses limited structure to interpret that interface for
counterfactuals or economically meaningful functionals, without committing to
one fully specified structural model.
```

Important refinement:

Do not define semistructural mainly as "anchored to an economic model." Define it as model-class robust:

```latex
\text{The object is valid across a class of models, not because one full model is true.}
```

## Kline: Most Useful Sections For The Semistructural Point

The key Kline passages for the project are:

1. Section 3.1.1, "Firm effects as restricted edge effects."
   Kline rewrites AKM as a restriction on directed mover wage changes:
   ```latex
   \Delta = B'\psi.
   ```
   The project opening is that the unrestricted edge effects \(\Delta_{jk}\) are the natural statistical interface, while AKM is one low-dimensional restriction on that interface.

2. Section 3.1.1 on cycles.
   Kline shows AKM is equivalent to saying that cycle effects are zero:
   ```latex
   \Delta = B'\dot\psi + C\dot\eta.
   ```
   The \(C\dot\eta\) component is where non-additivity, match effects, comparative advantage, or selection show up.

3. Section 3.2, "Evaluating the AKM restrictions."
   Kline finds AKM fits edge effects well but imperfectly. This is useful because the project does not need to reject AKM wholesale. It can say: AKM is a strong benchmark, but the residual cycle structure is economically meaningful and needs inference.

4. End of Section 3.2.
   Kline suggests model errors may reflect "heterogeneity in the firm effects faced by different sorts of workers." This is essentially the match-effect/comparative-advantage interpretation.

5. Section 3.3, "Causality."
   Kline separates causal mover wage changes from global AKM rankings. Edge effects can be causal under difference-in-differences-like assumptions, but turning them into a single firm ranking requires stronger assumptions.

6. Section 3.3.2 final paragraphs.
   Kline discusses weakening no-selection-on-treatment-effects, using dynamic propensity scores, and interactive factor models. He explicitly raises the question of whether the clustering step can be skipped and uncoarsened edge effects rationalized with factor models.

7. Section 4.2, cross-fitting and bias correction.
   KSS solves limited mobility bias for additive AKM. The project asks for an analogous inference layer once non-additive match effects are allowed.

8. Section 4.3, clustering approaches.
   This is the BLM contrast. Clustering buys tractability but coarsens firm heterogeneity and assumes groups of firms share exactly the same wage premium. The project pitch is "no clustering, graph-based inference."

## Kline Section 3.2 Writeup

```latex
\subsection*{Kline Section 3.2: Evaluating the AKM Restrictions}

\paragraph{Main finding.}
Kline's Section 3.2 shows that the AKM model is a useful but imperfect summary
of the wage changes workers experience when they move between firms. In the
Veneto data, AKM-predicted firm-effect differences explain a large share of the
variation in directed mover wage changes, but they do not explain all of it.
The remaining discrepancies are not just noise: after accounting for sampling
variation in edge-level wage changes, Kline finds residual model error. For the
present project, this is the central empirical motivation. AKM is close enough
to be a meaningful benchmark, but imperfect enough that the residual
match-specific component is worth modeling and doing inference on.

\paragraph{Technical explanation.}
Kline's starting object is the directed edge effect,
\[
    \Delta_{jk},
\]
the expected wage change for workers who move from firm \(j\) to firm \(k\).
The AKM model predicts that this object should be equal to the difference
between two firm effects:
\[
    \Delta_{jk} = \psi_k - \psi_j.
\]
Thus AKM does not merely estimate firm effects; it imposes restrictions on the
entire network of mover wage changes. In graph language, if \(B\) is the
firm-edge incidence matrix, AKM imposes
\[
    \Delta = B'\psi.
\]
This implies that wage changes around any cycle in the mobility graph must add
up consistently. For example, if workers move \(1 \to 2\), \(2 \to 3\), and
\(3 \to 1\), AKM requires
\[
    \Delta_{12}+\Delta_{23}+\Delta_{31}=0.
\]

Kline evaluates these restrictions by comparing estimated edge effects
\(\hat\Delta\) to AKM predictions \(\tilde\Delta\). A naive comparison is too
favorable to AKM because some edges are mechanically fit by the model,
especially bridge edges that just-identify firm effects. He therefore separates
actual model error from edge-level wage noise. The key decomposition is that a
residual sum of squares contains two pieces:
\[
    \text{residual variation}
    =
    \text{model error}
    +
    \text{noise in estimated edge effects}.
\]
After subtracting the expected noise component, Kline finds that AKM captures a
large share of true edge-effect variation, but not all of it.

This is exactly the semistructural opening. The object \(\Delta\) is a
statistical interface between the data and economic models of wage setting.
AKM corresponds to the special case in which all edge effects can be compressed
into firm effects. The residual cycle component is where match effects,
comparative advantage, or selection into different firm-worker pairings can
enter. The project should therefore frame Section 3.2 not as a rejection of AKM,
but as evidence that the AKM interface should be enlarged.
```

## Kline Section 3.3 Writeup

```latex
\subsection*{Kline Section 3.3: Causality}

\paragraph{Main finding.}
Kline's Section 3.3 shows that mover wage changes can have a causal
interpretation even when AKM additivity fails, but turning those causal edge
effects into a global ranking of firms requires much stronger assumptions. This
distinction is crucial for the project. The semistructural object should not
begin by assuming that every firm has one worker-invariant wage premium.
Instead, it can begin with causal edge or match effects and then ask which
additional structure is needed to summarize them.

\paragraph{Technical explanation.}
Kline treats a move from firm \(j\) to firm \(k\) like a many-treatment
difference-in-differences comparison. Let \(Y_{it}(j)\) denote worker \(i\)'s
potential wage at time \(t\) if employed at firm \(j\). Under assumptions
analogous to exclusion, parallel trends, and stationarity, the average wage
change of workers moving from \(j\) to \(k\) identifies the average causal
effect for that group of movers:
\[
    E[Y_{i2}-Y_{i1}\mid D_{i1}=j,D_{i2}=k]
    =
    \Delta_{jk}.
\]
The event-study designs in Card, Heining, and Kline fit here. They check
whether workers moving to high-wage firms were already on different wage trends
before the move. If pre-trends are flat, the observed wage jump is more
plausibly interpreted as a firm-related causal effect rather than as ordinary
worker wage growth.

But Kline emphasizes that causal edge effects are not automatically global firm
effects. The group of workers moving from \(j\) to \(k\) may differ from the
group moving from \(k\) to \(m\). If treatment effects are heterogeneous, then
the pairwise causal effects can be intransitive:
\[
    \Delta_{12} > 0,\qquad
    \Delta_{23} > 0,\qquad
    \Delta_{13} < 0.
\]
In that case, there may be no single firm ranking \(\psi_j\) such that
\[
    \Delta_{jk} = \psi_k-\psi_j
\]
for every pair of firms.

To recover a global AKM-style ranking, Kline introduces a stronger assumption:
no selection on treatment effects. This allows mobility to depend on average
firm quality, but rules out the idea that particular workers select into firms
because they have especially high match-specific gains there. Comparative
advantage and search models generally violate this condition. That is why this
section points directly toward the semistructural project: the empirically
credible causal object may be the edge or match effect, while the global firm
effect is an additional restriction.

This also clarifies the connection to Sorkin and Warwar. The older AKM event
study and separability tests asked whether additivity was a reasonable
approximation and often failed to reject it. Sorkin and Warwar's paired-mover
design, as described in the project note, suggests those tests were
underpowered: match effects can be a real share of firm-effect variation. In
Kline's language, this means the cycle restrictions may fail in economically
meaningful ways. The project can therefore be framed as building inference for
the causal and semistructural objects that remain once we stop forcing all
mover wage changes into a single additive AKM ranking.
```

## Notation Table

```latex
\subsection*{Notation}

\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
Symbol & Meaning \\
\hline
\(i\) & Worker index. \\
\(j,k,m\) & Firm indices. \\
\(t\) & Time period. In Kline's exposition, often \(t \in \{1,2\}\). \\
\(D_{it}\) & Firm employing worker \(i\) in period \(t\). \\
\(Y_{it}\) & Observed log wage of worker \(i\) in period \(t\). \\
\(Y_{it}(j)\) & Potential log wage of worker \(i\) in period \(t\) if employed at firm \(j\). \\
\(\alpha_i\) & Worker fixed effect in the AKM model. \\
\(\psi_j\) & Firm fixed effect or firm wage premium in the AKM model. \\
\(\varepsilon_{it}\) & Residual wage component in the AKM model. \\
\(\Delta_{jk}\) & Directed edge effect: average wage change for workers moving from firm \(j\) to firm \(k\). \\
\(\Delta\) & Vector collecting all directed edge effects in the mobility graph. \\
\(\hat{\Delta}\) & Estimated edge effects from observed mover wage changes. \\
\(\tilde{\Delta}\) & AKM-predicted edge effects, computed from estimated firm-effect differences. \\
\(B\) & Incidence matrix of the firm mobility graph. It maps firm effects into edge-level firm differences. \\
\(B'\psi\) & Vector of AKM-implied wage changes across directed firm-pair edges. \\
\(C\) & Matrix collecting cycle directions in the mobility graph. \\
\(\eta\) & Cycle-effect component; captures deviations from AKM additivity around cycles. \\
\(H\) & Projection or hat matrix mapping estimated edge effects into AKM-predicted edge effects. \\
\(h_{\ell\ell}\) & Leverage of edge \(\ell\); high leverage means the edge is mechanically fit more closely. \\
\(M = I-H\) & Residual-maker matrix for deviations between edge effects and AKM predictions. \\
\(n_\ell\) & Number of movers observed along edge \(\ell\). \\
\(u\) & Edge-level wage-change error; captures worker-level noise around edge effects. \\
\hline
\end{tabular}
\caption{Notation for Kline's AKM, edge-effect, and causality discussion.}
\end{table}
```

Additional semistructural extension notation:

```latex
\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
Symbol & Meaning in the semistructural extension \\
\hline
\(a_i\) & Worker-side latent or observed attribute relevant for match-specific wage gains. \\
\(b_j\) & Firm-side latent or observed attribute relevant for match-specific wage gains. \\
\(c(a_i,b_j)\) & Non-additive match component in wages. \\
\(\Lambda\) & Low-rank interaction matrix in a bilinear match-effect specification. \\
\(\beta_0\) & Scalar interaction parameter in the Tukey-style model \(c(a_i,b_j)=\beta_0 a_i b_j\). \\
\(\operatorname{Var}(\alpha)\) & Variance of worker effects. \\
\(\operatorname{Var}(\psi)\) & Variance of firm effects. \\
\(\operatorname{Cov}(\alpha,\psi)\) & Sorting of high-wage workers into high-wage firms. \\
\(\operatorname{Var}(c)\) & Variance contribution of match-specific or interaction effects. \\
\hline
\end{tabular}
\caption{Additional notation for the proposed semistructural non-additive model.}
\end{table}
```

## Borovickova-Shimer: Why ATT/ITT Instead Of AKM Variance Components

```latex
\subsection*{Why Borovi\v{c}kov\'a--Shimer Focus on ATT/ITT Rather Than AKM Variance Components}

\paragraph{Main point.}
Borovi\v{c}kov\'a and Shimer are asking a different question from the AKM
variance-decomposition literature. AKM asks how much of observed wage variation
can be summarized by worker effects, firm effects, and sorting:
\[
    \operatorname{Var}(\alpha_i), \qquad
    \operatorname{Var}(\psi_j), \qquad
    \operatorname{Cov}(\alpha_i,\psi_j).
\]
Borovi\v{c}kov\'a and Shimer instead ask what observed wages reveal about the
underlying production and selection process that generates accepted matches.
For that question, the natural distinction is between the wage or surplus effect
for all potential meetings and the wage or surplus effect for the selected
matches that actually form. This is why the relevant language is closer to
ITT versus ATT than to an AKM variance decomposition.

\paragraph{Technical explanation.}
In the AKM model, observed log wages are written as
\[
    w_{ij} = \alpha_i + \psi_j + \varepsilon_{ij}.
\]
The target parameters are second moments of the additive representation:
\[
    \operatorname{Var}(\alpha_i), \qquad
    \operatorname{Var}(\psi_j), \qquad
    \operatorname{Cov}(\alpha_i,\psi_j).
\]
These objects are useful summaries of observed wage inequality. They tell us
how dispersed worker effects are, how dispersed firm wage premia are, and
whether high-wage workers are sorted into high-wage firms.

Borovi\v{c}kov\'a and Shimer study a setting in which wages are generated by a
search and matching process. In that environment, a worker-firm pair is observed
only if the meeting is accepted. Let \(M_{ij}=1\) denote that worker \(i\) and
firm \(j\) actually form a match. The observed wage object is therefore not an
unconditional object over all possible worker-firm meetings. It is closer to
\[
    E[w_{ij} \mid M_{ij}=1],
\]
an average over accepted matches. By contrast, the corresponding object over all
potential meetings would be
\[
    E[w_{ij}].
\]
The difference between these two objects is the selection problem. Accepted
matches are not a random sample of all potential meetings. They are selected by
the equilibrium acceptance rule, which depends on match productivity, outside
options, and the distribution of match-specific shocks.

This is why the ATT/ITT distinction is useful. The treatment-on-the-treated
object asks about the effect for the selected matches that actually form:
\[
    E[w_{ij}(1)-w_{ij}(0)\mid M_{ij}=1].
\]
The intention-to-treat or potential-meeting object asks about the effect before
conditioning on acceptance:
\[
    E[w_{ij}(1)-w_{ij}(0)].
\]
The key point is that observed wages identify objects filtered through
selection. They need not reveal the unconditional production technology or the
wage surface over all possible worker-firm pairs.

This matters because the AKM firm effect \(\psi_j\) is not automatically a
structural firm productivity parameter. In a search model, observed wages
combine several ingredients:
\[
    \text{production complementarity}
    +
    \text{match-specific shocks}
    +
    \text{selection into accepted matches}
    +
    \text{wage setting or bargaining}.
\]
Therefore, even if observed wages admit an additive representation,
\[
    w_{ij} = \alpha_i + \psi_j + \varepsilon_{ij},
\]
that additivity may be a property of selected equilibrium wages, not a property
of the underlying production function. In Borovi\v{c}kov\'a--Shimer, additive
wages can arise as a special consequence of the selection mechanism. Moving
away from the special case reintroduces non-additivity.

\paragraph{Connection to the project.}
This distinction helps clarify the semistructural motivation. The AKM/KSS
literature targets second moments of an additive wage representation:
\[
    \operatorname{Var}(\alpha_i), \qquad
    \operatorname{Var}(\psi_j), \qquad
    \operatorname{Cov}(\alpha_i,\psi_j).
\]
Borovi\v{c}kov\'a--Shimer show that the economic meaning of these objects
depends on the selection process that generates observed matches. The proposed
project sits between these approaches. It keeps the AKM-style empirical
interface -- worker effects, firm effects, edge effects, and the mobility graph
-- but allows for non-additive match components:
\[
    w_{ij}
    =
    \alpha_i + \psi_j + c(a_i,b_j) + \varepsilon_{ij}.
\]
The goal is then to define and estimate economically meaningful functionals of
this richer wage surface, such as the variance contribution of match effects,
without committing to a fully specified search model. In this sense,
Borovi\v{c}kov\'a--Shimer explain why additivity is not innocuous, while the
semistructural project asks how to do inference once additivity is relaxed.
```

## Shimer-Smith Bellman Section: Correct Notation

Use the notation from the paper:

```latex
W(x) = \text{expected present value of an unmatched agent of type } x.
```

```latex
W(x\mid y) = \text{present value for type } x \text{ while matched with } y.
```

```latex
S(x\mid y) \equiv W(x\mid y)-W(x)
```

is the personal surplus from being matched.

The unmatched Bellman equation is:

```latex
rW(x)
=
\rho \int_{\mathcal M(x)} S(x\mid y)u(y)\,dy.
```

While unmatched, \(x\) earns zero flow payoff. At flow rate \(\rho u(y)dy\), she meets type \(y\). If \(y\in \mathcal M(x)\), the match is acceptable and she gains:

```latex
S(x\mid y)=W(x\mid y)-W(x).
```

There is no \(\delta\) in this equation because an unmatched agent has no match to be destroyed. The relevant hazard while unmatched is the meeting rate \(\rho\), not the separation rate \(\delta\).

The matched Bellman equation is:

```latex
rW(x\mid y)
=
\pi(x\mid y)-\delta S(x\mid y).
```

Here:

```latex
\pi(x\mid y)
```

is the endogenous matched flow payoff to type \(x\) when matched with \(y\). It is not \(W\).

The term

```latex
-\delta S(x\mid y)
```

is the expected capital loss from match destruction. At Poisson rate \(\delta\), the match ends, and \(x\) falls from \(W(x\mid y)\) back to \(W(x)\), losing:

```latex
W(x\mid y)-W(x)=S(x\mid y).
```

Clean split:

```latex
\underbrace{rW(x\mid y)}_{\text{return on asset value}}
=
\underbrace{\pi(x\mid y)}_{\text{flow payoff while matched}}
+
\underbrace{\delta[W(x)-W(x\mid y)]}_{\text{continuation / capital loss}}.
```

Important clarification:

```latex
W = present value / asset value,
\qquad
\pi = matched flow payoff,
\qquad
f = total match output,
\qquad
S = capital gain from moving from unmatched to matched.
```

The confusing part is that \(rW\) has units of a flow. But \(rW\) is not the primitive flow payoff. It is the annuity value of holding the asset \(W\). The primitive matched flow payoff is \(\pi(x\mid y)\), and the primitive unmatched flow payoff is zero.

With Nash bargaining, they impose:

```latex
S(x\mid y)=S(y\mid x)
```

and the resource constraint:

```latex
\pi(x\mid y)+\pi(y\mid x)=f(x,y).
```

Using the matched Bellman equation for both agents and the resource constraint gives:

```latex
S(x\mid y)
=
\frac{f(x,y)-rW(x)-rW(y)}{2(r+\delta)}.
```

Interpretation:

Personal surplus is half the excess of total match output over the two agents' flow values of remaining unmatched. The denominator \(r+\delta\) discounts for impatience and match destruction.

Project connection:

Shimer-Smith makes selection explicit. A match is observed only if both agents prefer matching to continuing search. Therefore observed wages/payoffs are not simply the primitive production function \(f(x,y)\). They are filtered through search, continuation values, mutual acceptance, and bargaining. This is the bridge to Borovickova-Shimer and to the semistructural project: additivity in observed wages may be a special equilibrium consequence of selection, not a primitive restriction on production.
```

