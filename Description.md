# H2 Producer Capacity Planning — Results Analysis (2026-08-06 run)

Analysis of `outputs/plan_final_artifact_data.json`, the latest `plan_h2_capacity.py`
run: all 13 eligible countries, no budget cap, 1 representative day/month, joint
shared demand pool of 100 MW/hr (`--joint-pool-mw 100`), electrolyser required
alongside any other asset (`require_electrolyser_for_others`), downstream demand
flexible ±20% around each country's own baseline with net-zero enforced **per
representative day** (24h cycle), unlimited grid/H2 exchange, CAPEX/lifetime sourced
from `Help/Candidates (Edited).docx`'s 2030 candidate catalog. Converged in 4
iterations, 247.1s. Total system CAPEX €2.854B, best objective (annualized CAPEX + 1yr
operating cost) **-€51.57M** (net profit).

This document explains *why* the model landed on the specific capacity mix it did —
which countries got more electrolyser, which got batteries, and which got no storage
at all — grounded in the actual run's numbers, not just the abstract mechanism.

## Headline result

| Country | Electrolyser | Wind | PV | Battery | Tank | Raw CAPEX | Pool share |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL | **100.0** | 100 | 100 | 0 | 20 | €262.5M | **68.0%** |
| SK | 5.0 | 100 | 100 | **50.0** | **50** | €304.6M | 3.1% |
| LU | 5.0 | 100 | 100 | 10.0 | 50 | €258.9M | 3.4% |
| PL | 5.0 | 100 | 100 | 10.0 | 20 | €207.8M | 3.1% |
| AT, BE, CZ, DE, FR, HU, RO | 5.0 | 100 | 100 | 0 | 20 | €202.2M | ~2.8-3.4% |
| **HR, SI** | 5.0 | 100 | 100 | 0 | 20 | €202.2M | **0.0%** |

Three things immediately jump out and need explaining: (1) wind/PV are identically
maxed out in **every single country**, (2) electrolyser is bimodal — one country at
100 MW, everyone else pinned at the 5 MW floor, with two of those getting zero use out
of it, and (3) only 3 of 13 countries build a battery, at two different sizes, with no
obvious single reason why those three.

## 1. Why wind and PV are maxed out *everywhere*, with zero country-specific logic

Every one of the 13 countries builds the top catalog candidate for both wind (100 MW)
and PV (100 MW) — completely uniform, no variation at all. Looking at the catalog
itself explains why:

| Asset | EUR/MW at every candidate tier |
|---|---|
| Wind | ~1,328k–1,330k/MW at 5/10/50/100 MW — **flat, no economies of scale** |
| PV | exactly 500k/MW at 5/25/50/100 MW — **flat, no economies of scale** |

Unlike the electrolyser (below), wind and PV cost the *same* per MW regardless of
size — so there's no cost incentive from scale one way or the other. What actually
drives them to the ceiling is that **grid exchange is unlimited** in this pipeline
(`grid_cap_mw=inf`, §4.1) and **there's no budget cap** (`--no-budget`): any MWh a
turbine or panel produces beyond what the local electrolyser needs can always be sold
to the grid (`x_grid`) or credited via the green-certificate mechanism (`gc_sell`) at
that zone's own proxy price, with no connection-capacity ceiling to ever choke that
off. With CAPEX flat-per-MW and the revenue side effectively unconstrained, the model
has no reason *not* to build the maximum candidate everywhere — wind/PV sizing in this
run carries essentially zero country-specific signal; it's saturated by construction
of the run's own assumptions (unlimited exchange + no budget), not by each country's
actual solar/wind resource or price profile.

## 2. Why electrolyser is bimodal: one big hub (NL) instead of thirteen small ones

Two constraints force *every* country in this run to build at least *some*
electrolyser: `require_electrolyser_for_others` (wind/PV/tank can't be built without a
positive electrolyser candidate) means every country that builds anything — which, per
§1, is all 13 — must also build electrolyser; and the catalog's cheapest candidate
(5 MW, €5.21M) is the obvious way to satisfy that at minimum cost for a country that
has no other reason to go bigger.

But the pool needs real capacity behind it: `min_total_electrolyser_mw = pool_mw / η =
100 / 0.68 ≈ 147.06 MW` must be installed *in aggregate* across all included countries
for the shared 100 MW/hr demand pool to be physically servable at all (Formulation.md
§4.4 "Min-electrolyser" / §4.7). Twelve countries sitting at the 5 MW floor only
supplies 60 MW — nowhere near enough. Someone has to go further up the candidate
ladder. And the electrolyser catalog, unlike wind/PV, has **real economies of scale**:

| Candidate | EUR/MW | vs. 5 MW candidate |
|---|---:|---:|
| 5 MW | 1,042k/MW | — |
| 20 MW | 925k/MW | 11% cheaper |
| 50 MW | 772k/MW | 26% cheaper |
| 100 MW | 655k/MW | **37% cheaper** |

Given that, the cheapest way to close a ~87 MW aggregate shortfall is *one* big jump
to the 100 MW candidate (€65.5M total) rather than several countries each taking a
smaller step (e.g. four countries at 20 MW would cost 4×€18.5M = €74M for only 80 MW,
still short of the floor and *more* expensive). The master concentrates the shortfall
into a single country — NL — rather than spreading it, purely because concentration is
cheaper under this cost curve.

**NL's electrolyser runs flat-out.** Its realized downstream demand averages 68.0 MW —
*exactly* its own hourly ceiling (100 MW × 0.68 η = 68 MW) — meaning NL absorbs 68% of
the entire shared pool, every hour, at full utilization. Every other 5 MW-electrolyser
country is *also* running near its own ceiling (5 × 0.68 ≈ 3.4 MW, and their realized
demand sits at 2.8–3.4 MW) — so utilization rate isn't the story anywhere; every built
electrolyser in this solution runs close to 100% of what it physically can. The only
difference between NL and everyone else is scale, driven by the aggregate pool floor
plus economies of scale picking one concentrated winner.

*Why NL specifically, and not e.g. DE or FR?* All 13 countries face an *identical*
CAPEX catalog, so the tie-break is each country's own trained price-proxy economics
(§1). This run's data doesn't isolate one clean number that makes NL obviously best
(its own electricity price, €54.4/MWh, is close to several other countries', not
uniquely the cheapest — FR is lower at €44.0/MWh yet stayed at the floor). Benders
decomposition finds *a* good solution, not necessarily the unique global optimum
(Formulation.md §4.7 "LP dual non-uniqueness" / §4.6 "best FOUND, not proven
optimal") — the specific choice of *which* country becomes the pool's hub is
plausibly a near-tie the iterative cut sequence happened to settle on, not something
with a single deterministic cause visible in this run alone.

**HR and SI built an electrolyser they never use.** Both sit at the mandatory 5 MW
floor — required to also build their (maxed-out) wind/PV/tank — but their *realized*
downstream demand is **0.0 MW**. Their electrolysers are 100% idle, all year. This
traces to a real, already-documented quirk of the underlying price model: HR00 and
SI00 are the two CORE zones with no strongly price-correlated neighbour (see the
price-model's own "Key findings" — the neighbour-price feature, normally the dominant
driver almost everywhere else, has nothing strong to key off for these two). In this
run their H2 proxy price averages **~€114-116/MWh-H2**, versus **~€59/MWh-H2**
virtually everywhere else — essentially double. At that price regime it's never
worthwhile for the shared pool to route any demand to HR/SI's electrolysers, even
though the `require_electrolyser_for_others` constraint still forces them to pay
€5.21M each for an electrolyser purely as the "cost of admission" for their otherwise-
profitable wind/PV/tank build (§1/§4 below).

## 3. Why H2 tank storage is built almost everywhere, at one dominant size

11 of 13 countries build **exactly 20 MW** of tank; the other 2 (LU, SK) build 50 MW;
none build zero. The tank catalog explains the near-universal 20 MW choice directly —
its EUR/MW is **not** monotonic with size:

| Candidate | MWh | EUR/MW |
|---|---:|---:|
| 1 MW | 16.7 | 950k/MW |
| 5 MW | 166.7 | 750k/MW |
| **20 MW** | **666.7** | **700k/MW ← cheapest** |
| 50 MW | 3,333 | 1,300k/MW |

20 MW is the catalog's genuine cost "sweet spot" — cheaper per MW than both the
smallest *and* the largest candidate. So once a country decides tank storage is worth
building at all, 20 MW is the obvious default choice purely from the discrete cost
curve, independent of that country's own price signal — which is exactly the pattern
observed. Combined with the **unlimited H2 pipeline connection** (`h2_cap_mw=inf`),
tank + pipeline forms a fairly generic buy-low/sell-high H2 arbitrage position, and
every country in this run shows a broadly similar H2 price spread (σ ≈ €17-24/MWh-H2,
day range ≈ €35-47/MWh-H2 — see the per-country table below) — enough, apparently, to
make that 20 MW "default" position worthwhile almost everywhere. LU and SK go further,
to the 50 MW/3,333 MWh candidate (1,300k/MW, nearly double the 20 MW tier's cost per
MW) — consistent with both being otherwise the two most storage-heavy, highest-CAPEX
countries in the whole run (see §4).

## 4. Why only 3 countries build a battery — and why the pattern isn't clean

This is the one asset where a single tidy explanation genuinely doesn't hold up
against the data, and it's worth saying so plainly rather than forcing a story onto
it.

The battery catalog's EUR/MW roughly **doubles** between the two smaller candidates
(2/10 MW, 2h duration, ≈564k/MW) and the two larger ones (20/50 MW, 4h duration,
≈1,029k/MW) — because doubling the duration also doubles the MWh delivered per MW.
So "build battery at all" is really a bet on whether 2h of daily electricity-price
arbitrage covers the CAPEX + round-trip losses (92% efficiency), and "go to 4h instead
of 2h" is a second, separate bet that the *extra* duration earns back its roughly 2×
higher per-MW cost.

Checking each country's own average **intraday** electricity-price swing (max − min,
per representative day, averaged across the year) against what actually got built:

| Country | Avg. intraday swing (EUR/MWh) | Battery built |
|---|---:|---|
| **PL** | **90.6** | 10 MW (2h) |
| NL | 65.4 | — none — |
| DE | 61.7 | — none — |
| **LU** | 61.6 | 10 MW (2h) |
| BE | 61.6 | — none — |
| CZ | 54.3 | — none — |
| FR | 53.6 | — none — |
| HU | 36.3 | — none — |
| AT | 43.9 | — none — |
| SI | 31.5 | — none — |
| **SK** | **32.2** | **50 MW (4h)** |
| HR | 25.2 | — none — |
| RO | 25.2 | — none — |

PL's battery is easy to explain — it has by far the largest intraday price swing of
any country in the run (€90.6/MWh, ~40% above the next-highest), a textbook arbitrage
signal. LU's build is directionally consistent (well above the median). But the
pattern breaks down from there: **DE and NL have comparable-or-larger swings than LU
and built nothing**, while **SK builds the single largest, longest-duration battery in
the whole run despite one of the *lower* swings** (€32.2/MWh — barely above HR/RO,
which build none).

Rather than force a single-variable story, the honest reading is that battery/tank are
specifically the two assets Formulation.md already flags as the weakest-behaved part
of this pipeline:

- **§4.5 "LP dual non-uniqueness"** — battery/tank Benders cut coefficients are the
  ones most exposed to degenerate-optimum imprecision (≈0.1-0.7% relative error,
  empirically validated by finite difference), so a genuinely small underlying profit
  difference between "build" and "don't" can tip the discrete decision either way
  depending on which optimal vertex the LP solver happened to land on.
- **§4.6/§4.7** — Benders here is gap-tolerance-terminated on a MILP with a *discrete*
  candidate grid, returning the **best solution found**, not a certified global
  optimum. Near-tied battery decisions (and SK's jump straight to the 4h/50 MW tier
  rather than testing 20 MW first) are plausible outcomes of the specific cut sequence
  the 4-iteration run happened to generate, not necessarily what an exhaustive search
  would produce.

So: price volatility is *part* of the story (PL, and to a lesser extent LU, fit it
cleanly) but is not sufficient on its own to explain SK or the DE/NL non-builds — the
discrete, degeneracy-prone nature of this specific decision is a real, documented
characteristic of the model, not a data-analysis gap.

## Per-country reference data

| Country | p_elec avg | p_elec intraday swing | p_h2 avg | p_h2 intraday swing | GC buy (MWh) | GC sell (MWh) |
|---|---:|---:|---:|---:|---:|---:|
| AT | 66.3 | 43.9 | 59.4 | 41.0 | 0 | 331,201 |
| BE | 56.3 | 61.6 | 59.1 | 40.6 | 0 | 352,846 |
| CZ | 70.2 | 54.3 | 59.3 | 40.8 | 0 | 326,117 |
| DE | 60.0 | 61.7 | 59.8 | 35.1 | 0 | 349,339 |
| FR | 44.0 | 53.6 | 58.8 | 40.5 | 0 | 373,554 |
| HR | 78.8 | 25.2 | **115.7** | 36.5 | 0 | 416,989 |
| HU | 75.8 | 36.3 | 59.0 | 40.8 | 191 | 273,541 |
| LU | 56.4 | 61.6 | 59.8 | 41.3 | 229 | 355,735 |
| **NL** | 54.4 | 65.4 | 59.0 | 40.8 | **87,820** | 170,785 |
| PL | 80.9 | 90.6 | 58.9 | 40.5 | 117 | 354,086 |
| RO | 79.6 | 25.2 | 59.6 | 39.6 | 200 | 390,384 |
| SI | 77.1 | 31.5 | **113.4** | 47.1 | 0 | 264,027 |
| SK | 71.8 | 32.2 | 58.5 | 39.0 | 134 | 278,938 |

NL's green-certificate buy volume (87,820 MWh) dwarfs everyone else's (next highest is
HU at 191 MWh) — direct confirmation that NL's electrolyser load structurally exceeds
its own renewable generation by a huge margin (it's importing grid electricity at
scale to run a 100 MW electrolyser fed by the same 100+100 MW wind/PV every other
country has), consistent with NL operating as a genuine bulk hydrogen production hub
for the shared pool rather than a locally-self-sufficient producer.

## Caveats

- **CAPEX/lifetime are real, cited figures** (`Help/Candidates (Edited).docx`'s 2030
  column) but still one document's assumptions, not procurement-grade vendor quotes
  (Formulation.md §4.7).
- **Representative-day sampling** (1 day/month, 12 of 364 days) means every number
  above is a weighted *estimate*, not an exact full-year value (§2.6).
- **No budget cap and unlimited exchange** in this specific run are deliberate
  configuration choices (both pipeline defaults as of 2026-08-06), not universal
  truths — they're a large part of *why* wind/PV saturate everywhere and why NL's
  electrolyser economics look as extreme as they do; a budgeted or capacity-capped run
  would likely spread capacity very differently.
- **Battery/tank sizing carries the most solver-dependent noise** of the five assets
  (§4 above) — treat the specific battery on/off pattern as illustrative of the
  model's behavior under this configuration, not as a robust country-level ranking.
- Storage cycles (battery **and** the new demand-flex `shift`) are both **24h,
  per-representative-day, independent** — no cross-day carryover for either.
