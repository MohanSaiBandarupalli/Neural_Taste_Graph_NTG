# Startup strategy (hypotheses, not facts)

Status: **no customers, no pilots, no revenue, no partnerships.** Everything below is a plan to be
validated. Market claims are qualitative. Sizing must be built bottom-up from customer-discovery data, not
quoted from reports we have not verified.

## 1. Thesis

Large streaming services have big in-house recommendation teams. The many mid-sized catalogue businesses
(regional OTT services, niche SVOD, AVOD/FAST channels, publishers with video libraries, audiobook and
podcast apps, edtech video libraries) usually rely on generic vendor recommenders or simple rules. Our
hypothesis is that they would pay for three things those tools rarely combine:

1. **Journey-level optimisation** with fatigue awareness, and a simulator to check a policy *before*
   shipping it.
2. **Viewer-understandable taste models** (TasteTwin), which double as a trust and regulatory asset.
3. **Honest decision tooling** for non-ML executives: labelled evidence, scenario arithmetic, and a
   grounded analyst.

## 2. Ideal customer profile (to be validated)

* A subscription or ad-supported catalogue of 2k–200k titles with 50k–5M monthly active users.
* A recommendation team of 0–3 ML engineers.
* Retention or engagement owned by a product or growth lead who has budget.
* Willing to share anonymised interaction logs for a 6–8 week pilot.

## 3. Competitive landscape (public, qualitative)

| Category | Examples | Their strength | Where NTG may differ |
|---|---|---|---|
| Cloud recommender APIs | AWS Personalize, Google Recommendations AI | scale, easy procurement | journey simulation, fatigue modelling, viewer-facing taste profile, labelled evidence |
| Personalisation / CDP platforms | Algolia Recommend, Bloomreach, Dynamic Yield, Coveo | e-commerce depth, merchandising tooling | entertainment-specific taste dynamics, long-horizon policies |
| Media-specific vendors | several OTT recommendation vendors | domain integrations (CMS, players) | open evaluation methodology, OPE, simulator |
| Build in-house | open-source (RecBole, Merlin, implicit) | full control | turnkey product and analytics around the models |

We have not benchmarked any competitor. Differentiation claims are hypotheses for customer interviews.

## 4. Product tiers (pricing hypotheses to test, not established prices)

| Tier | Includes | Pricing hypothesis to test |
|---|---|---|
| Developer | sandbox, synthetic data, rate-limited API | free |
| Startup | hosted models on the customer catalogue, TasteTwin, explanations | usage-based (per 1k recommendation calls) |
| Business | + enterprise analytics, journey simulator, experiments | platform fee + usage |
| Enterprise | private deployment, SSO, custom models, SLA, data residency | annual contract |

Pricing will be tested with Van Westendorp-style questions and pilot conversion, not guessed.

## 5. Customer discovery plan (next 8 weeks)

1. Build a list of 40 target companies matching the ICP.
2. Run 20+ interviews with product, growth or data leads. Goal: test the problems, not pitch.
   * How do you choose what to recommend today? How do you evaluate a change?
   * Have you seen repetition or fatigue complaints? How would you measure them?
   * Would a pre-launch policy simulator change your decisions? What would make you trust it?
   * Who owns the budget? What does a successful pilot look like?
3. Kill criteria: if fewer than 25% of interviewees rank the journey or fatigue problem in their top 3,
   pivot to the strongest problem we did hear.
4. Offer 2–3 design-partner pilots: offline evaluation on their logs (free), then an online A/B test with
   a pre-registered primary metric.

## 6. Pilot strategy

* Weeks 1–2: data ingestion and validation (`/v1/enterprise/data/validate`), temporal split, baselines on
  the customer's data.
* Weeks 3–4: offline evaluation report; fatigue sensitivity estimated from their logs, which is the key
  unknown behind the journey policy's value.
* Weeks 5–8: A/B test (greedy-fresh against the orchestrator, or their current policy against the NTG
  hybrid) with guardrails.
* Deliverable: a pre-registered result, published to them whether positive or negative.

## 7. Roadmap to revenue (product)

Real-data benchmark (ML-32M) → CMS / player event connectors → per-tenant model training jobs → billing
integration (Stripe-ready usage table exists) → SSO → SOC 2 readiness.

## 8. Major risks

| Risk | Mitigation |
|---|---|
| The journey or fatigue gain does not hold on real data (it is shown only in simulation) | Estimate fatigue from logs first. The product still provides a strong hybrid recommender, explanations and analytics. |
| Hybrid ranking ≈ tuned baseline (seen on synthetic data) | Position on product and tooling value. Improve candidate recall. Validate on ML-32M and pilot data. |
| Buyers prefer cloud vendors for procurement reasons | Offer a "bring your own cloud" deployment and integrate as a policy layer on top of their recommender. |
| Data-sharing friction and privacy | On-prem or VPC deployment, data processing agreement, minimal data (IDs, timestamps, ratings or watch events). |
| Founder bandwidth | Narrow ICP and one design partner at a time. |
| IP | Get an attorney opinion before public disclosure (see ROADMAP "IP"). The public repository already discloses the implementation. |
