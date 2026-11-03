# Product specification: NTG v4

## Problem

Catalogue-heavy subscription services face three recurring recommendation failures:

1. **Static taste models.** Profiles average a viewer's whole history, so a change in taste takes months
   to show up.
2. **Myopic optimisation.** Each list is optimised for the next click. Viewers see the same titles and
   genres again and again, which researchers associate with fatigue and disengagement.
3. **Opaque decisions.** Neither viewers nor operators can see *why* something was recommended, or what a
   policy change would do before shipping it.

## Users and jobs

| User | Job to be done | NTG surface |
|---|---|---|
| Viewer | "Find something I'll enjoy tonight, without endless scrolling" | Home rows, discovery modes, NL discover, title detail with "why" |
| Viewer | "Understand and steer what the service thinks I like" | TasteTwin page, feedback, consent, export/delete |
| Viewer | "Break out of a rut" | Exploratory / unexpected modes, hidden interests, journeys |
| Recommendation / ML team | "Compare models and policies honestly before an A/B test" | Performance, model lab, journey simulator, OPE |
| Product / growth lead | "Which audiences are drifting, fatigued or at risk, and what would an intervention be worth?" | Audience, fatigue, risk, business simulator |
| Content / programming | "Which parts of the catalogue are under-discovered?" | Content intelligence |
| Executive | "One page with honest numbers" | Executive dashboard, AI analyst |
| Platform customer (B2B) | "Call an API with my user's history and get explained recommendations" | `/v1/b2b/recommend` with API keys and usage metering |

## Functional requirements (all implemented in v4)

* R1. Personalised recommendations with four exploration levels and grounded explanations.
* R2. Natural-language intent that separates explicit, inferred and unsupported constraints.
* R3. TasteTwin profile: long-term and short-term taste, emerging and declining interests, stability,
  exploration willingness, confidence, uncertainty, and evolution over time. Updates on every feedback
  event without retraining.
* R4. Hidden interests with evidence; multi-session journey plans.
* R5. Feedback: watched, rated, like, dislike, not interested ("not interested" removes the title from
  recommendations).
* R6. Enterprise: segments, TasteTwin analytics, model and policy evaluation, simulator, fatigue,
  engagement-risk proxy, business scenarios, content intelligence, experiments, monitoring, data
  validation, AI analyst, admin.
* R7. Multi-tenant SaaS foundations: organisations, RBAC, API keys, usage, audit.
* R8. Privacy: consent, export, deletion.

## Non-functional requirements

| Requirement | Target | Status |
|---|---|---|
| Recommendation latency (engine, single node) | p95 < 50 ms | 8.3 ms without explanations, 11.4 ms with (measured) |
| API p95 at 1 concurrent request | < 100 ms | 20.8 ms (measured) |
| Reproducible build | one command | `ntg build` |
| Honest evidence | every number labelled and traceable | enforced in the UI and docs |

## Explicit non-goals for v4

Video playback, payments, real subscription data, LLM-generated explanations, and multi-region
deployment.

## Success metrics for a pilot (proposed, to be agreed with the customer)

* Primary: long-horizon satisfaction proxy, such as 28-day return rate and titles completed per active
  week, in an A/B test against the customer's current policy.
* Guardrails: session-1 engagement (must not fall beyond the agreed margin), catalogue coverage,
  complaint rate.
* Model: offline NDCG / recall on the customer's own temporal holdout before any online test.
