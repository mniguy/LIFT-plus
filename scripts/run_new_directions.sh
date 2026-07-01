#!/bin/bash
#
# Directions alpha (online ESS-gated) and beta (structured/grouped) prior adaptation.
# Both are post-hoc on the saved test logits dumped by run_test_agnostic.sh -- no
# retraining, runs in minutes. Used to decide which direction (if either) to pursue.
#
# Decision rules:
#   alpha WINS if, at small batch B, the ESS gate ('ess'/'floor') stays >= no-adapt while
#         the KL-only gate ('kl') drops below no-adapt (over-correction); at large B all
#         converge to the transductive em_shrink number.
#   beta  WINS if em_group_shrink keeps the [1] All win, SHRINKS the [2] Few/Many crater
#         vs em_shrink, and shows lower [3] tail prior-L1 error than full em.

PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-"imagenet_lt places_lt"}
TAU=${TAU:-1.0}
GAMMA=${GAMMA:-1.0}

for ds in ${DATASETS}; do
  ROOT="output/test_agnostic/${ds}/lift+"
  echo ""
  echo "######################## ${ds} ########################"

  echo "==== alpha: online ESS-gated (mode=batch, sweep B) ===="
  ${PYTHON} scripts/online_prior_adapt.py --root "${ROOT}" \
    --batch-sizes 50 200 1000 5000 --mode batch --tau "${TAU}" --gamma "${GAMMA}" --kappa 500

  echo "==== alpha: online ESS-gated (mode=cumulative, sweep B) ===="
  ${PYTHON} scripts/online_prior_adapt.py --root "${ROOT}" \
    --batch-sizes 50 200 1000 5000 --mode cumulative --tau "${TAU}" --gamma "${GAMMA}" --kappa 500

  echo "==== beta: structured / grouped prior estimation ===="
  ${PYTHON} scripts/structured_prior_adapt.py --root "${ROOT}" --tau "${TAU}" --gamma "${GAMMA}"
done
