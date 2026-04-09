# Theoretical Framework

## Motivation

The 2022–2024 wave of tech-sector layoffs presents a natural laboratory for
studying how financial markets process signals about corporate efficiency and
technological change. Unlike ordinary restructuring events, these layoffs were
publicly framed in the context of a specific technological shift — the rapid
commercialization of generative AI following the November 2022 launch of ChatGPT.

This creates a rare opportunity to test two competing theories about how
markets interpret workforce reductions, and whether the *stated rationale*
for a layoff affects market pricing beyond the simple fact of the layoff itself.

---

## Three Competing Hypotheses

### H1 — Restructuring Signal Hypothesis

**Theory:**
Corporate layoffs are a costly signal of managerial discipline (Jensen 1986,
Weiss & Nikitin 1998). By reducing headcount, management credibly commits to
lower operating costs and higher future cash flows. Markets reward this signal
regardless of the stated reason. Under this hypothesis, the AI narrative is
economically irrelevant — what matters is that the firm is demonstrating fiscal
discipline.

**Mechanism:**
Layoff → Reduced operating cost base → Higher expected future margins →
Higher equity value

**Testable prediction:**
- CAAR[−1,+1] ≥ 0 for layoff announcements in general
- β₁ (ai_mentioned) ≈ 0: the AI label adds no incremental price impact
- β₃ (DID estimator ai × post_chatgpt) ≈ 0: the ChatGPT era does not change
  the market's reaction to AI-framed layoffs

---

### H2 — Labor-for-AI Substitution Hypothesis

**Theory:**
When a company announces that it is replacing labor with AI/automation tools,
investors update their beliefs about the firm's future productivity trajectory.
This channel was not credible before generative AI existed at scale; after
ChatGPT's launch, the technology became legible and tractable enough for
investors to assign meaningful probability to genuine productivity gains.

**Mechanism:**
AI-framed layoff → Capital-labor substitution believed → Higher future
productivity per employee → Higher expected revenue / profit per dollar of cost →
Premium equity valuation

This hypothesis predicts a *structural break* at the ChatGPT launch: the
same AI language in a layoff announcement should carry more weight post-2022
than pre-2022, because investors now have a concrete reference point for what
AI-driven productivity looks like.

**Testable predictions:**
- β₁ (ai_mentioned) > 0: AI-framed layoffs receive a market premium
- β₃ (DID) > 0: this premium is *larger* in the post-ChatGPT era
- Effect should be stronger in long windows [0,+20], [−5,+60] than [−1,+1],
  because efficiency gains are slow to materialize (medium-run drift)

---

### H3 — Over-hiring Correction Hypothesis

**Theory:**
Tech companies systematically over-hired during the 2020–2021 low-interest-rate
boom (Zuckerberg publicly acknowledged Meta "hired too many people"). The
subsequent layoffs are a mechanical correction of this error, not a forward-
looking strategic shift. Under this view, the AI narrative is strategic
communication — a reframing designed to make necessary cost-cutting palatable
to investors and employees. The economic content is the same as any other
restructuring; only the language differs.

**Mechanism:**
Over-hiring correction → Return to normal operating efficiency →
Modest positive market reaction (cost savings), but no premium for the
AI label because it carries no incremental information about future productivity

**Testable predictions:**
- CAAR[−1,+1] could be positive or small-negative (consistent with cost savings
  being partially anticipated)
- β₁ (ai_mentioned) ≈ 0: the AI label carries no incremental signal
- β₃ (DID) ≈ 0: the ChatGPT launch does not structurally change investor
  interpretation of AI-framed layoffs
- CAR is positively correlated with layoff magnitude (% workforce cut) —
  consistent with "the more over-hiring is reversed, the more value is recovered"
  — but NOT with the AI label

---

## Mapping Hypotheses to Empirical Tests

| Test | What it tests | Supports H1 | Supports H2 | Supports H3 |
|------|---------------|-------------|-------------|-------------|
| CAAR[−1,+1] significant | Any announcement effect | Positive | Positive | Small positive / negative |
| β₁ (ai_mentioned) in OLS | AI label premium | ≈ 0 | > 0 | ≈ 0 |
| β₃ (DID: ai × post) | ChatGPT structural break | ≈ 0 | > 0 (stronger post-ChatGPT) | ≈ 0 |
| R² in cross-section | Market efficiency | Low (1–2%) | Moderate | Low |
| Calendar-time α | Clustering-robust drift | Positive | Positive, larger for AI | Positive, similar for AI vs non-AI |
| Repeat events | Information decay | Smaller for later events | Smaller for later events | Similar across sequences |
| Pre-announcement drift | Information leakage | Negligible | Negligible | Negligible |

---

## Discussion of Null Results

Under market efficiency (Fama 1970), all three hypotheses predict low R² in
the cross-section of announcement-day CARs. Even if H2 is correct (the AI
label carries genuine information), this information should be absorbed
*rapidly* into prices. By the time of the announcement, news articles about
AI strategy and workforce reduction have typically been circulating for days
or weeks. Therefore, the announcement day is not the moment of information
revelation; much of the price response may occur over a longer window, or
even before the formal announcement.

The key empirical question is therefore not just "is β₁ significant on day 0?"
but rather:
1. **Does the medium-run CAAR** [0,+20] differ between AI and non-AI layoffs
   (H2 predicts yes, H1/H3 predict no)?
2. **Is there a structural break at the ChatGPT launch** in how AI-framed
   layoffs are priced (H2 predicts yes)?
3. **Is the DID result robust** to placebo breakpoints, alternative AI
   definitions, and exclusion of clustering-heavy periods?

Our null DID result (β₃ insignificant in most specifications) is most
consistent with H1 or H3. The short-run negative CAAR (−0.49% to −1.2%,
significant) combined with positive medium/long-run drift (+2–6%) is
consistent with H1: markets initially treat the layoff as a negative shock
(talent loss, business disruption) but subsequently update positively as
cost savings materialize. This pattern does not require AI to be the driver.

---

## References

- Fama, E.F. (1970). Efficient capital markets. *Journal of Finance*.
- Jensen, M.C. (1986). Agency costs of free cash flow. *American Economic Review*.
- Weiss, L.A. & Nikitin, G.S. (1998). Performance of companies that delete
  employees. Working paper, Tulane University.
- Datta, S., Iskandar-Datta, M., & Raman, K. (2001). Executive compensation
  and corporate acquisition decisions. *Journal of Finance*.
- Acemoglu, D. & Restrepo, P. (2018). The race between man and machine.
  *American Economic Review*.
