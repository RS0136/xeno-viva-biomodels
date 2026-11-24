# xeno-biomodels

SBML models of BioModels on which five ODE engines agree, run through
[viva-biomodels](https://github.com/vivarium-collective/viva-biomodels) (Vivarium 2.0), and beside each of them, in
the same composite of [process-bigraph](https://github.com/vivarium-collective/process-bigraph), an organism that is
orthogonal to it. Orthogonality is measured, with the definition written out below, from the series of the runs,
for the organism as a whole and layer by layer.

What the runs give is in three reports, with their figures:

* [`runs/report.md`](runs/report.md): each SBML organism with its xeno
* [`runs/community.md`](runs/community.md): all the pairs in one vessel
* [`runs/studies.md`](runs/studies.md): what those readings depend on

A manuscript that draws on the three is in [`paper/`](paper/README.md).

In short, with the plan as it is in [`configs/experiment.yaml`](configs/experiment.yaml):

* 171 models of the viva-biomodels corpus are on the plateau: all five engines ran them, and no variable differs
  between any two engines by more than 1e-5 of its size. Three of them are models of the metabolism of a
  microorganism on a nutrient and are the receivers: BIOMD0000000172 (*S. cerevisiae*, glycolysis),
  BIOMD0000000217 (*E. coli*, ammonium assimilation) and BIOMD0000000017 (*L. lactis*, pyruvate metabolism).
  Their scores are 3.3e-6, 3.7e-6 and 4.9e-6.
* An organism that lives on a substrate of its own in the same vessel (design `xeno`) is orthogonal to each of the
  three at every stage of the definition, at every layer and for both kinds of protected quantity: D = 0, O = 1,
  stages 1, 3 and 4 hold, D_removal = 0. The zero is exact: every stored number of the SBML organism is the same in
  every run, bit for bit.
* The same organism with a transporter that also takes the nutrient of the receiver, at 0.001 to 0.3 of its
  specificity for its own substrate, is not: D rises from 0.11–1.7 to 17–18. Where it lands differs between the
  models: at the level of the nutrient for the yeast and the *L. lactis* model, at the adenylylation of glutamine
  synthetase for the *E. coli* model, where the quantities that every organism has say "tolerable" (D = 0.79) and
  the one by which the organism is known says that it is not (1.74).
* With the three pairs poured into one vessel, interference follows the species that the organisms have in common
  and nothing else. The two SBML organisms that live on glucose compete, and one is washed out. Xenos with
  substrates of their own move nobody. Xenos on one common substrate drive each other out and still move no SBML
  organism. And when each xeno, orthogonal to its own partner, takes a hundredth of the nutrients of the other
  pairs, a xeno reaches its own partner through the others: D of the xeno of *E. coli* to *E. coli* is 2.4, in a
  pair that is exactly orthogonal alone.
* The studies put these readings against what they could depend on. The zero survives everything that was tried:
  every assumed number of the world at a quarter to four times its value, every choice of the measure, every
  coupling interval, a second species to share, 18 worlds of other SBML models taken by rule, and an activity that
  changes in time. The size of D of a control depends on the assumptions; a comparison of steady states would not
  see most of it; steps of the activity are not the worst that a sender can do within the same bounds; and the
  layers of an organism with one way in are fewer than two independent readings, not six.
* Further studies look at what a value of D is made of. The function of the sender does not bound its
  activity from above, so D is a value for a range of activities. With a shared nutrient it is the leak times
  the capacity that a step adds to the population of the sender, times a gain that has a closed form in the
  response coefficients of the receiver alone: 18, 270 and 15 for the three receivers, against 17 to 18, 241 to
  263 and 15 in the runs (`activity`). Inside the organism D is the dose-response of the SBML model alone,
  followed along the course of the medium, to within 1.5 % (`quasistatic`). Tolerances derived from the growth
  function are a guarantee and not a verdict: of twelve leaky senders the 10 % convention passes two, the box of
  guaranteed tolerances five, and the function itself is kept with nine (`functional`). Every world with a leaky
  sender, written as a single system of equations and integrated by CVODE and by LSODA, gives the D of the
  composite to 0.06 % (`monolith`). Two pairs that coexist stably show a xeno reaching its own partner through
  the other pair, with a niche overlap of zero (`coexistence`). A noisy record of a receiver still says which
  activity a leaky sender chose where every tolerance is kept (`information`). And a model of a growing cell,
  which computes its own growth rate, takes the place of the growth law of the world: the closed form holds, and
  the inside of that receiver is as slow as the vessel, so it departs from the model alone by up to 11 % (`cell`).

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/). viva-biomodels is a workspace and not a package, so it is cloned
beside this code, and its simulator stack is installed from an export of its own lock file.

```bash
git clone https://github.com/vivarium-collective/viva-biomodels.git third_party/viva-biomodels
```

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python --no-deps -r env/viva-biomodels.lock.txt
```

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

```bash
uv pip install --python .venv/bin/python pytest
```

`env/viva-biomodels.lock.txt` is `uv export --frozen --no-dev --no-emit-project --no-hashes` of the `uv.lock` of
viva-biomodels at commit `d3d11a2`, which is the commit this was run with. `--no-deps` matters: the lock pins
`viva-superpowers` by commit while one of its packages asks for the branch, and uv refuses to resolve the two.

libroadrunner links against `libpython3.12.so.1.0`, which an interpreter managed by uv keeps off the path of the
loader; `import xeno_biomodels` loads it first. To use Tellurium outside this package, set `LD_LIBRARY_PATH` to
the `lib` directory of the interpreter.

## Run

```bash
.venv/bin/xvb all -j 12
```

```bash
.venv/bin/xvb studies -j 12
```

```bash
.venv/bin/xvb figures
```

| Command | What it does | Writes |
| --- | --- | --- |
| `xvb select` | Scores the agreement of the engines on every model of the corpus, lists the plateau, and screens it for models that take a species up from their surroundings | `runs/selection/` |
| `xvb verify` | Runs the receivers through the `compare-simulators` composite of viva-biomodels under COPASI, Tellurium and simbio, and sets the result against the corpus | `runs/verify/` |
| `xvb worlds -j N` | Every receiver with every design of the sender: baseline, steps of the activity, removal; then the measure, whole and layer by layer. `--analyze` measures again from the runs that are there | `runs/worlds/` |
| `xvb community -j N` | The pairs poured into one vessel, for every community design: its fate with nothing stepped, every step of every xeno, every removal; then the measure for every ordered pair of organisms. `--analyze` as above | `runs/community/` |
| `xvb studies -j N` | The fifteen studies below (`--only NAME ...` for some of them) | `runs/studies/` |
| `xvb figures` | The figures and the three reports | `runs/figures/`, `runs/report.md`, `runs/community.md`, `runs/studies.md` |
| `xvb paper` | The figures and the tables of the manuscript, from the runs and the reports (`--only fig3 ...` for some figures) | `paper/figures/`, `paper/tables/` |
| `xvb all -j N` | select, verify, worlds, community, figures | |
| `xvb orthogonality` | The measure on any series files | what `--out` names |

`xvb select` fetches the SBML file of every model of the plateau on its first run (about 20 minutes; they are kept
under `models/`). `xvb all` then takes about eight minutes on 12 cores, six of them for the communities, and
`xvb studies` about forty for the first eight studies and about half an hour more for the others, most of it
for the community of `coexistence`, which is followed for five hundred residence times before anything is
stepped. The study `cell` needs the SBML file of BioModels entry MODEL1502270000 under `models/MODEL1502270000/`;
where the loader of viva-biomodels does not fetch it, download it by hand:

```bash
mkdir -p models/MODEL1502270000
```

```bash
curl -L -o models/MODEL1502270000/MODEL1502270000_url.xml "https://www.ebi.ac.uk/biomodels/model/download/MODEL1502270000?filename=MODEL1502270000_url.xml"
```

```bash
.venv/bin/python -m pytest -q
```

The measure alone, on two runs of a world, with two groups of protected quantities read beside the whole:

```bash
.venv/bin/xvb orthogonality runs/worlds/BIOMD0000000217/leak-1e-3/baseline/protected.csv --activity runs/worlds/BIOMD0000000217/leak-1e-3/u=8/protected.csv --protect 'medium.NH4=rel:0.1' --protect 'B.species.AMP=rel:0.1' --protect 'B.growth_rate=fn' --function B.growth_rate --qmin rel:0.5 --allocation dependent --receiver-density B.density --viable 1e-6 --group 'common=medium.NH4,B.growth_rate' --group 'specific=B.species.AMP'
```

## The definition, as `xeno_biomodels.orthogonality` computes it

A sender A and a receiver B. A baseline run and runs in which one activity of A is stepped at time zero, all from
the same initial conditions and under the same supply. A run is a CSV file: `time` and one column per quantity.

**Protected quantities and tolerances.** B protects y_k (k = 1..N), columns of the files, each with a tolerance δ_k:

| written | δ_k |
| --- | --- |
| `abs:X` | X |
| `rel:R[,floor:F]` | max(R \|y_k(0)\|, F), with y_k(0) of the baseline at time zero |
| `fn` | margin / (m max_t \|∂Q_B/∂y_k\|) |

Q_B is the function that B must keep, an expression of its protected quantities, with Q_B ≥ Q_min. The margin is
what the baseline has to spare: min_t Q_B − Q_min (`pointwise`), Q_B(T) − Q_min (`terminal`) or the mean of Q_B over
time − Q_min (`time_average`). m is N (`all`) or the number of protected quantities that Q_B depends on
(`dependent`). The slopes are central differences along the baseline; their largest absolute value over the stored
times is taken.

**Lemma.** To first order, whenever |Δy_k| ≤ δ_k for every k,

    |ΔQ_B| ≤ Σ_k |∂Q_B/∂y_k| |Δy_k| ≤ Σ_k max_t |∂Q_B/∂y_k| δ_k

With every tolerance from `fn` the right side is (N / m) margin: the margin itself when m = N, and also with
`dependent`, where the quantities that Q_B does not depend on contribute nothing. So D ≤ 1 implies Q_B ≥ Q_min in
the sense of the mode, to first order. This is why a tolerance `fn` is not arbitrary. For any mixture of tolerances
the measure reports the right side over the margin (`bound_over_margin`); the implication holds when it is at most 1.

**Loss.** τ_B(u) is the first stored time at which the density of B is below `viable`, or an earlier one at which
Q_B < Q_min when the function is read pointwise. Read at the end or on average, the function is judged on the whole
run: its loss is put at the horizon and every stored time is read.

**Admissible activities.** U_A is the baseline u_0 and the runs in which A keeps its own function, Q_A ≥ Q_A,min.
A run that does not is left out (`excluded_by_sender_function`). The removal of A is outside U_A and read apart
(`D_removal`).

**Interference.**

    D(t) = sup over u in U_A with t < τ_B(u) of max_k |y_k^u(t) − y_k^u0(t)| / δ_k
    D = sup_t D(t),    O = 1 / (1 + D)

u_0 is always a row of zeros, so D(t) ≥ 0. O is for display; the manuscript does not use it.

**Stages.** 1 trajectory: no loss and D ≤ `stage_tol` (1e-6). 3 tolerable: no loss and D ≤ 1. 4 distribution: no
loss and the upper bound on the bits ≤ `bits_tol`. (The manuscript calls the last the information criterion; the
study `information` computes the bits themselves, by sampling the noise, where the measure gives two bounds.)

**Bits.** The response of an activity is a point: its changes over the tolerances, (N, T) flattened; the baseline is
the origin. With observation noise σ on each of the M = N T numbers, the upper bound is
(M / 2) log2(1 + P / (M σ²)) with P the largest squared norm of a point, and the lower bound is that of a uniform
choice through the pairwise Bhattacharyya distances, −mean_i log2 mean_j exp(−|x_i − x_j|² / (8 σ²)).

**Groups.** The protected quantities may be put into groups (`groups` of `measure`, `--group` of the command). A
group is read exactly as the whole is, on its own quantities alone and with the tolerances that they have in the
whole, so D of the whole is the largest D of the groups that cover it. The losses of the receiver are those of
the whole: a receiver is not lost at one layer and working at another.

### Where the definition is silent, and what the program does

* A quantity counts as one that Q_B depends on when the peak of its slope, times the size of the quantity, is more
  than 1e-6 of the size of Q_B. The step of a central difference is 1e-6 of the size of the quantity.
* `fn` on a quantity that Q_B does not depend on is an error, not an infinite tolerance.
* `rel` of a quantity that is zero at time zero needs a floor; without one it is an error.
* Q_min written `rel:R` is R times Q of the baseline at time zero, for B and for A.
* The time average is the trapezoid over the stored times.
* Q_B is read only while the density is at or above `viable`. A receiver that is lost on the baseline itself is not
  measured (`not_functional`).
* With a loss in any admissible run the bits are not given and stage 4 fails. σ is 1 and `bits_tol` 1e-3 unless given.
* Runs must be stored on the times of the baseline. A number that is missing while the receiver works is an error,
  not a change of zero.
* A run in which the SBML model stops having a solution counts as a loss of the receiver: it is read up to its
  last stored time, and its density is put at zero from the next (`layers.write_collapsed`). This happens in the
  yeast model, which has no steady state below 0.52 of the glucose of its file; leaving such runs out would take
  the strongest interference out of the reading.
* Besides the definition, the result names the run and the quantity at which D is attained (`D_activity`,
  `D_quantity`).
* There is no stage 2 in the definition as given, and none is computed.

## What an organism protects

`xeno_biomodels.layers` reads `protect` of the plan. An organism protects quantities of two kinds, each at a layer:

* **common**: the same quantity for every organism, one at each layer, so that organisms can be set side by side
* **specific**: what is characteristic of one organism, the pool and the step by which its metabolism is known

| layer | what an SBML model of metabolism holds there | common quantity |
| --- | --- | --- |
| medium | what the organism lives on, in the vessel | the level of its nutrient |
| particle | energy conversion | the ATP that its reactions form per cell volume and time |
| molecule | its metabolites | the sum of their pools (carriers and states of proteins apart) |
| complex | its enzymes at work, and the states of its proteins | its uptake of the nutrient |
| individual | its growth | its growth rate (tolerance `fn`) |
| population | its density | its density |

The specific quantities of the three receivers are acetaldehyde and pyruvate decarboxylase (yeast), glutamine,
glutamine synthetase and its adenylylated form (*E. coli*), pyruvate and lactate dehydrogenase (*L. lactis*). An
SBML model of metabolism has nothing at the levels of the atom, the supramolecule (a genome), the condensate or the
organelle, and nothing is read there; an organism of laws has no `particle` and no `molecule` either.

The measure is read once for all of it and, with the same tolerances, for every layer, for each kind, and for each
kind at each layer (`runs/worlds/layers.csv`, `runs/community/layers.csv`). Which quantity stands for a layer, and
which quantities count as characteristic, are choices of the plan; the study `settings` reads every species and
every rate of a model alone to show what the choice does.

## The world

`xeno_biomodels.organisms` has the processes of process-bigraph, and `xeno_biomodels.world` wires them:

    supply      dilution rate and feed: one node, read by the chemostat and by both organisms
    medium      the concentrations in the vessel, shared by everything
    B           the SBML organism: the Tellurium process of viva-tellurium with the interface of an organism
    A           the organism of laws (Michaelis-Menten uptake, growth by a constant yield with maintenance)
    chemostat   dS/dt = k (S_feed - S)

B is `SBMLOrganism`, a subclass of the `TelluriumProcess` that viva-biomodels registers and wires into its
`biomodel-process` composite, configured by `viva_biomodels.simulators.process_config`. The SBML model is not
changed. Before each interval the nutrient is read from the medium into the boundary species of the model; after
it, what the reactions of the model took is returned to the medium, times the density of the organism; and the
organism grows at μ = k J / J_rest, with J its uptake of the nutrient. The feed and the yield are not chosen: the
feed is the one that holds the vessel at the value of the SBML file when B is alone at the steady state of its
model, and the yield the one at which it grows at the dilution rate there. That steady state is therefore the rest
state of B in the world; it matches the state that COPASI reaches on the untouched SBML file to 1e-12.

Everything that the world adds is an assumption and is marked so in the plan: the chemostat, the exchange, the
growth law, the dilution rate, the densities, and every value of A.

Every exchange is a rate times the interval, so a state in which the rates balance stays where it is whatever the
interval. The organisms are coupled once per interval (0.25 min) by summing what each adds to the medium; within
an interval the SBML model is integrated by CVODE. The plan is written in minutes; a model is run in its own unit
of time, and the times of the plan are turned into it.

A design of the sender is the leak of its transporter for the nutrient of the receiver, relative to its own
substrate (a number, as in the plan), or `{share: f, of: species}`: a leak in the units of no model,
half-saturated at the level of the SBML file and sized so that the sender takes at rest the share f of what the
receiver takes of the species. The studies that go beyond the three receivers of the plan use the second form.

A receiver need not be a pathway model. A model may hold its nutrient as a parameter and not as a boundary
species (`uptake_reactions` then names the reactions whose rate is its uptake), and it may compute a growth rate
of its own (`growth_variable`), at which the organism then grows in the place of the law above; the level of the
nutrient at which that rate is the dilution rate is found from the model. The viable density of the plan may be
written `rel:x`, a share of the density at the start of the baseline, for models whose densities are in units of
their own. The study `cell` uses all three.

## Several pairs in one vessel

`xeno_biomodels.community` puts every pair of the plan, an SBML organism and its xeno, into one composite. Each pair
is first brought to rest in a vessel of its own, and the vessels are poured together in equal volumes: every
density and every concentration becomes the mean over the vessels, the feed the mean of the feeds, the dilution
rate stays. Nothing is chosen for the community beyond what the pairs had. A species that two SBML models hold is
one species of the common medium, in the unit of the medium: `GLCo` of the yeast model and `halfglucose` of the
*L. lactis* model, which counts glucose in halves, are both `glucose`.

Three designs differ in what a xeno has in common with the organisms of the other pairs; in all of them it takes
nothing of its own partner.

| design | the xenos |
| --- | --- |
| `private` | each on a substrate and a product of its own |
| `shared` | all on one common substrate (the pairs as they were built) |
| `partner-only` | as `shared`, and each takes the nutrients of the SBML organisms of the other pairs at 0.01 |

For each design the poured community is followed with nothing stepped (who stays), and then, from its state 6000
min after the pouring, with every xeno stepped and every organism removed. The measure is applied to every ordered
pair of organisms: D from the steps of a xeno, D_removal from taking the sender away. `runs/community/<design>/
matrix.csv` has one row per ordered pair, and with it whether the sender has a way to the receiver at all: a chain
of species that one organism writes and the next reads. The exact zeros of the runs are the ordered pairs with no
such chain.

## The studies

`xeno_biomodels.studies` puts the readings against what they could depend on. Each study is a module that writes
tables under `runs/studies/<name>/`; `runs/studies.md` is written from them.

| study | what is varied | what is read |
| --- | --- | --- |
| `sensitivity` | every assumed number of the world at 0.25, 0.5, 2 and 4 times its value, two other growth laws, the horizon; in the community the cross specificity (0.001 to 0.1) and the dilution rate | the zero of `xeno`, D of a control and the layer at which it lands |
| `settings` | every choice of the measure (tolerance, Q_min, allocation, modes, viable density, noise, stored times), from the runs that are there | D and the stages; the lemma on every stored time; every species and rate of a model read alone |
| `numerics` | the coupling interval (a quarter to twice) and the tolerances of the integrator | what the method leaves in D |
| `channels` | where the interference enters: the sender takes the nutrient of the receiver, or another species that its model reads | the layers, by the one way in and by the other |
| `cascade` | nothing: the runs of the plan and of `channels` | when the interference is at each layer, and how many independent readings the layers are |
| `receivers` | the SBML model: every model of the plateau that reads a species that it takes up, by rule | the quantities common to all organisms, layer by layer |
| `schedules` | the activity in time: square waves of five periods, and the switching that the step responses say is worst | D at each layer against the time scale of the activity |
| `comparison` | nothing: other indices on the same worlds | niche overlap (Pianka), the linear response of the steady state, the steady states of the steps, beside D |
| `activity` | nothing: every step of the plan read alone, and the closed form of the rest state and of the linear response of the vessel | D against the capacity that a step adds to the population of the sender; the gain in closed form beside the runs; the activity at which the tolerance is reached, and the one above which the receiver has no rest state |
| `quasistatic` | nothing: the SBML model alone, with its surroundings held (`xeno_biomodels.isolated`) | its dose-response and response coefficients; the level below which it has no steady state; D of the model alone followed along the medium of every run, beside D of the run |
| `functional` | the tolerances: every quantity that the growth function reads gets the tolerance that the function gives it | D with tolerances of first order and with those that keep the function over the whole box (checked to be monotone, so that its corners decide); the same counting only changes that lower the function, and with a conserved pair counted once; the share of its margin that the function itself lost |
| `monolith` | the method: every world with a leaky sender as a single system of ODEs (the SBML model with the world as rate rules), under CVODE and under LSODA (COPASI, run in an interpreter of its own by `copasi_course`) | D and D(t) beside the composite |
| `coexistence` | the community: two pairs whose SBML organisms live on different nutrients and whose xenos have substrates of their own, so that the four coexist stably (`private`; `overlapping`, each xeno also on the substrate of the other; `indirect`, each also taking the nutrient of the other pair without growing on it); both leaks of `indirect` at other values | the measure for every ordered pair, step by step, the niche overlap beside it, and how far the baseline still moves |
| `information` | the noise: from a hundredth of a tolerance to a thousand, and every 2nd to 64th stored time | the bits that a noisy record of the receiver carries about which run it was, between the two bounds of the measure |
| `cell` | the receiver: the model of a growing cell of Weiße et al. (2015; BioModels MODEL1502270000), which computes its own growth rate | the measure, and `activity`, `quasistatic` and `functional` on its runs |

The series of a world that a study runs are not kept, only what was measured (`cell` keeps them); `settings`,
`cascade`, `comparison`, `activity`, `quasistatic`, `functional` and `information` can be run again from what is
there, the others run their worlds again. The first eight studies are reported in `runs/studies.md`; the others
write their tables under `runs/studies/<name>/` and are reported in the manuscript.

## Layout

| Path | What |
| --- | --- |
| `configs/experiment.yaml` | The plan: selection, receivers, what is protected, supply, sender and designs, activities, time, measure, community, studies |
| `src/xeno_biomodels/corpus.py`, `screen.py` | Agreement of the engines over the corpus; which models have an environment |
| `src/xeno_biomodels/verify.py` | The receivers through viva-biomodels, against its corpus |
| `src/xeno_biomodels/organisms.py`, `world.py` | The processes and the composite |
| `src/xeno_biomodels/layers.py` | What an organism protects: common and specific quantities, layer by layer |
| `src/xeno_biomodels/experiment.py` | The runs of every world and their measure |
| `src/xeno_biomodels/community.py` | The pairs in one vessel: its runs and the measure of every ordered pair |
| `src/xeno_biomodels/orthogonality.py` | The measure |
| `src/xeno_biomodels/studies/` | The fifteen studies; the figures and the report of the first eight |
| `src/xeno_biomodels/isolated.py` | The SBML model of a receiver alone: steady states at given surroundings, and its rate laws at a given state |
| `src/xeno_biomodels/figures.py`, `report.py`, `community_report.py` | Figures and the two reports of the runs |
| `env/viva-biomodels.lock.txt` | The simulator stack, pinned |
| `third_party/viva-biomodels/` | The checkout of viva-biomodels, untouched |
| `models/` | SBML files fetched from BioModels |
| `runs/` | Everything the commands write |
| `src/xeno_biomodels/paper/` | The figures (`figures.py`, PDF) and the tables (`tables.py`, LaTeX) of the manuscript, and what they share (`style.py`) |
| `tests/` | Unit tests of the measure, the processes, the studies and the tables, on cases with a known answer |
| `paper/` | The manuscript for BioSystems: `main.tex`, `supplement.tex`, `references.bib`, `highlights.tex`, and what `xvb paper` writes; see `paper/README.md` |

## Limits

They are listed with their evidence in section 5 of each report and at the end of `runs/studies.md`. Those that
bound the main statements:

* The score of a model is agreement between engines on the same SBML file, not agreement with an experiment.
* Orthogonal means orthogonal to the SBML model in this world. B can be reached only through what the world lets
  vary; every other boundary species of a model stays at the value of its SBML file, and what a model does not
  have (pH, temperature, space) cannot carry interference at all.
* The layers are a reading of the model, not of the cell. Which quantity stands for a layer is a choice, quantities
  within a layer differ by an order of magnitude, and in a world with one way in the layers are fewer than two
  independent readings: D of a layer is not an independent test of the organism.
* The size of every D that is not an exact zero depends on the assumed numbers of the world.

## License

The code (`src/`, `tests/`, `configs/`, `env/`) is under the Apache License 2.0 (`LICENSE`), the license of
process-bigraph, bigraph-schema, viva-biomodels, Tellurium and libRoadRunner, on which it is built. The other
libraries that it imports are under permissive licenses (BSD for NumPy and pandas, the Matplotlib license, MIT
for PyYAML, antimony and simbio, Apache 2.0 for pyarrow) or are used as installed libraries without being
copied or changed (libSBML, LGPL 2.1; COPASI and basico, Artistic 2.0); none of them is distributed here.

The SBML and SED-ML files under `models/` are entries of BioModels (https://www.ebi.ac.uk/biomodels/), copied
unchanged. BioModels places its models in the public domain under CC0 1.0; the original publications are named
in each file and should be cited with the model.

