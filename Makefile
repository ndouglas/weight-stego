# Reproduce every number in the report from committed files. No API calls unless you run an `api-*` target.
# Requires uv (https://docs.astral.sh/uv/). MNIST and Fashion-MNIST download to data/ on first use.
PY = uv run python

.PHONY: reproduce setup verify tables zoo baseline
reproduce: setup verify tables baseline   ## ~15 min on a laptop CPU; writes results/*.md

setup:
	uv sync --frozen

verify:     ## retrain all clean nets from seed and compare bit-for-bit; check every prompt matches its weights
	$(PY) src/verify_nets.py

tables:     ## score the committed raw API results (runs/api) -> results/api.md; payload statistics -> results/payload_stats.md
	$(PY) src/score_api.py
	$(PY) src/payload_stats.py

zoo:        ## 180 nets (6 recipes x seeds 100-129) for the non-LLM baseline; ~12 min; git-ignored
	cd src && $(PY) zoo.py

baseline: zoo   ## logistic-regression and single-rule readers on the zoo -> results/baseline.md
	$(PY) src/baseline.py

# ---- Paid: re-running the LLM reads (Anthropic Message Batches, 50% price). Needs ANTHROPIC_API_KEY in .env. ----
# Each submit prints a batch id; then: $(PY) src/api_reads.py collect <id>, and append the id to runs/api/batches.txt.
# These are the exact submissions behind runs/api/ (order matters: later batches supersede earlier custom_ids).
.PHONY: api-estimate api-confirmatory api-effort-grid api-opus-max api-graft
api-estimate:
	$(PY) src/api_reads.py estimate
api-confirmatory:   ## 92 reads, Opus + Fable, high effort, bare prompts (~$30); truncated Fable reads rerun with --ids ... --max-tokens 128000
	$(PY) src/api_reads.py submit --sets backbone,rung,camouflage,launder
api-effort-grid:    ## 15 Fable reads on the camouflage nets (~$36); truncated ones rerun with --max-tokens 300000 --beta output-300k-2026-03-24
	$(PY) src/api_reads.py submit --models fable --sets camouflage --cells high:num,max:bare,max:num --max-tokens 128000
api-opus-max:       ## 5 Opus reads (~$7)
	$(PY) src/api_reads.py submit --models opus --sets camouflage --cells max:num --max-tokens 300000 --beta output-300k-2026-03-24
api-graft:          ## 40 reads, both models, 2 repeats (~$19)
	$(PY) src/api_reads.py submit --sets graft --repeats 2 --max-tokens 128000
