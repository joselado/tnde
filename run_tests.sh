#!/bin/bash
# Run all gates on one core (the machine is shared).
cd "$(dirname "$0")"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
CORE=${CORE:-5}
fail=0
for t in tests/test_dense_breathing.py tests/test_tt.py tests/test_fourier.py \
         tests/test_kinetic.py tests/test_fit.py tests/test_evolve1d.py \
         tests/test_evolve2d.py tests/test_reference_r30.py; do
  echo "=== $t"
  if taskset -c $CORE python3 "$t" 2>&1 | grep -v "WARNING.*xla_bridge"; then :; else fail=1; fi
done
exit $fail
