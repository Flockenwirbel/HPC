#!/bin/bash
set -e

cd "$(dirname "$0")"

# Load environment
spack load cuda
spack load gcc@10.2.0

echo "============================================"
echo "       Benchmark: test_gmem (STRIDE)        "
echo "============================================"
echo ""
printf "%-10s %-15s\n" "STRIDE" "Bandwidth(GB/s)"
printf "%-10s %-15s\n" "------" "----------------"

for STRIDE in 1 2 4 8; do
    # Modify STRIDE in a temp file and compile
    sed "s/^#define STRIDE [0-9]*/#define STRIDE $STRIDE/" test_gmem.cu > test_gmem_tmp.cu
    nvcc test_gmem_tmp.cu -o test_gmem -O2 -code sm_60 -arch compute_60
    # Run and capture output
    output=$(srun --exclusive --gres=gpu:1 ./test_gmem 2>&1)
    # Extract bandwidth
    bandwidth=$(echo "$output" | grep "bandwidth:" | awk '{print $2}')
    printf "%-10s %-15s\n" "$STRIDE" "$bandwidth"
done
rm -f test_gmem_tmp.cu

echo ""
echo "============================================"
echo "   Benchmark: test_smem (BITWIDTH x STRIDE) "
echo "============================================"
echo ""
printf "%-12s %-10s %-15s\n" "BITWIDTH" "STRIDE" "Bandwidth(GB/s)"
printf "%-12s %-10s %-15s\n" "--------" "------" "----------------"

for BITWIDTH in 2 4 8; do
    for STRIDE in 1 2 4 8 16 32; do
        # Modify BITWIDTH and STRIDE in a temp file and compile
        sed -e "s/^#define BITWIDTH [0-9]*/#define BITWIDTH $BITWIDTH/" \
            -e "s/^#define STRIDE [0-9]*/#define STRIDE $STRIDE/" \
            test_smem.cu > test_smem_tmp.cu
        nvcc test_smem_tmp.cu -o test_smem -O2 -code sm_60 -arch compute_60
        # Run and capture output
        output=$(srun --exclusive --gres=gpu:1 ./test_smem 2>&1)
        # Extract bandwidth
        bandwidth=$(echo "$output" | grep "bandwidth:" | awk '{print $2}')
        printf "%-12s %-10s %-15s\n" "$BITWIDTH" "$STRIDE" "$bandwidth"
    done
done
rm -f test_smem_tmp.cu

echo ""
echo "Done!"