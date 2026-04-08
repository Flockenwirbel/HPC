#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 解析数据
data = defaultdict(lambda: {'naive': {}, 'shared_memory': {}})

with open('res.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        mode = parts[0]
        block_x = int(parts[1])
        block_y = int(parts[2])
        exec_time = float(parts[4])
        
        key = (block_x, block_y)
        data[key][mode] = exec_time

# 按 block_x 和 block_y 分组
by_x = defaultdict(lambda: {'x': [], 'naive': [], 'shared': [], 'speedup': []})
by_y = defaultdict(lambda: {'y': [], 'naive': [], 'shared': [], 'speedup': []})

for (bx, by), modes in sorted(data.items()):
    naive_time = modes.get('naive', 0)
    shared_time = modes.get('shared_memory', 0)
    
    if naive_time > 0 and shared_time > 0:
        speedup = naive_time / shared_time
        
        by_x[bx]['x'].append(by)
        by_x[bx]['naive'].append(naive_time)
        by_x[bx]['shared'].append(shared_time)
        by_x[bx]['speedup'].append(speedup)
        
        by_y[by]['y'].append(bx)
        by_y[by]['naive'].append(naive_time)
        by_y[by]['shared'].append(shared_time)
        by_y[by]['speedup'].append(speedup)

# 创建图表
fig = plt.figure(figsize=(20, 15))

# 1. 性能随线程块总大小 (BlockX * BlockY) 的变化
ax1 = plt.subplot(2, 3, 1)
block_sizes = []
naive_times = []
shared_times = []
for (bx, by), modes in data.items():
    if 'naive' in modes and 'shared_memory' in modes:
        block_sizes.append(bx * by)
        naive_times.append(modes['naive'])
        shared_times.append(modes['shared_memory'])

ax1.scatter(block_sizes, naive_times, alpha=0.5, label='Naive', marker='o')
ax1.scatter(block_sizes, shared_times, alpha=0.5, label='Shared Memory', marker='x')
ax1.set_xlabel('Total Threads per Block (BX * BY)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Performance vs. Block Size')
ax1.legend()
ax1.grid(True)

# 2. 不同 Block X 对应的平均执行时间
ax2 = plt.subplot(2, 3, 2)
sorted_bxs = sorted(by_x.keys())
avg_naive_by_bx = [np.mean(by_x[bx]['naive']) for bx in sorted_bxs]
avg_shared_by_bx = [np.mean(by_x[bx]['shared']) for bx in sorted_bxs]
x_indices = np.arange(len(sorted_bxs))
width = 0.35
ax2.bar(x_indices - width/2, avg_naive_by_bx, width, label='Naive')
ax2.bar(x_indices + width/2, avg_shared_by_bx, width, label='Shared')
ax2.set_xticks(x_indices)
ax2.set_xticklabels(sorted_bxs, rotation=90)
ax2.set_xlabel('Block X')
ax2.set_ylabel('Avg Execution Time (ms)')
ax2.set_title('Impact of Block X')
ax2.legend()
ax2.grid(True, axis='y')

# 3. Speedup 热力图 (Shared Memory 带来的提升)
ax3 = plt.subplot(2, 3, 3)
block_xs = sorted(by_x.keys())
all_bys = set()
for bx in block_xs:
    all_bys.update(by_x[bx]['x'])
sorted_bys = sorted(list(all_bys))
heatmap_speedup = np.zeros((len(block_xs), len(sorted_bys)))

for i, bx in enumerate(block_xs):
    bx_data = data_by_bx_by = {(bx, by): modes for (bx_alt, by), modes in data.items() if bx_alt == bx}
    for j, by in enumerate(sorted_bys):
        modes = bx_data.get((bx, by))
        if modes and 'naive' in modes and 'shared_memory' in modes:
            heatmap_speedup[i, j] = modes['naive'] / modes['shared_memory']

im3 = ax3.imshow(heatmap_speedup, cmap='RdYlGn', aspect='auto')
ax3.set_xlabel('Block Y')
ax3.set_ylabel('Block X')
ax3.set_title('Speedup Heatmap (Naive / Shared)')
ax3.set_xticks(range(len(sorted_bys)))
ax3.set_xticklabels(sorted_bys, rotation=90)
ax3.set_yticks(range(len(block_xs)))
ax3.set_yticklabels(block_xs)
plt.colorbar(im3, ax=ax3, label='Speedup Factor')

# 4. 固定 Block X = 32 时，Block Y 的影响 (Warp 效率相关)
ax4 = plt.subplot(2, 3, 4)
if 32 in by_x:
    ax4.plot(by_x[32]['x'], by_x[32]['naive'], 'o-', label='Naive (BX=32)')
    ax4.plot(by_x[32]['x'], by_x[32]['shared'], 's-', label='Shared (BX=32)')
    ax4.set_xlabel('Block Y')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Impact of Block Y (for BX=32)')
    ax4.legend()
ax4.grid(True)

# 5. Naive 性能热力图 (重新对齐坐标)
ax5 = plt.subplot(2, 3, 5)
heatmap_naive_ref = np.zeros((len(block_xs), len(sorted_bys)))
for i, bx in enumerate(block_xs):
    for j, by in enumerate(sorted_bys):
        modes = data.get((bx, by))
        if modes and 'naive' in modes:
            heatmap_naive_ref[i, j] = modes['naive']
im5 = ax5.imshow(heatmap_naive_ref, cmap='YlOrRd', aspect='auto')
ax5.set_xticks(range(len(sorted_bys)))
ax5.set_xticklabels(sorted_bys, rotation=90)
ax5.set_yticks(range(len(block_xs)))
ax5.set_yticklabels(block_xs)
plt.colorbar(im5, ax=ax5, label='Time (ms)')
ax5.set_title('Naive Time Heatmap')

# 6. Shared Memory 性能热力图 (重新对齐坐标)
ax6 = plt.subplot(2, 3, 6)
heatmap_shared_ref = np.zeros((len(block_xs), len(sorted_bys)))
for i, bx in enumerate(block_xs):
    for j, by in enumerate(sorted_bys):
        modes = data.get((bx, by))
        if modes and 'shared_memory' in modes:
            heatmap_shared_ref[i, j] = modes['shared_memory']
im6 = ax6.imshow(heatmap_shared_ref, cmap='YlGnBu', aspect='auto')
ax6.set_xticks(range(len(sorted_bys)))
ax6.set_xticklabels(sorted_bys, rotation=90)
ax6.set_yticks(range(len(block_xs)))
ax6.set_yticklabels(block_xs)
plt.colorbar(im6, ax=ax6, label='Time (ms)')
ax6.set_title('Shared Memory Time Heatmap')

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to performance_analysis.png")

# 生成统计信息
print("\n" + "="*60)
print("PERFORMANCE ANALYSIS SUMMARY")
print("="*60)

all_naive = []
all_shared = []
all_speedups = []

for (bx, by), modes in data.items():
    if 'naive' in modes and 'shared_memory' in modes:
        all_naive.append(modes['naive'])
        all_shared.append(modes['shared_memory'])
        all_speedups.append(modes['naive'] / modes['shared_memory'])

print(f"\nNaive Mode:")
print(f"  Average: {np.mean(all_naive):.4f} ms")
print(f"  Min:     {np.min(all_naive):.4f} ms")
print(f"  Max:     {np.max(all_naive):.4f} ms")

print(f"\nShared Memory Mode:")
print(f"  Average: {np.mean(all_shared):.4f} ms")
print(f"  Min:     {np.min(all_shared):.4f} ms")
print(f"  Max:     {np.max(all_shared):.4f} ms")

print(f"\nSpeedup (Naive / Shared Memory):")
print(f"  Average Speedup: {np.mean(all_speedups):.2f}x")
print(f"  Min Speedup:     {np.min(all_speedups):.2f}x")
print(f"  Max Speedup:     {np.max(all_speedups):.2f}x")

# 找最优配置
best_config_naive = min(data.items(), key=lambda x: x[1].get('naive', float('inf')))
best_config_shared = min(data.items(), key=lambda x: x[1].get('shared_memory', float('inf')))

print(f"\nBest Configurations:")
print(f"  Naive:        Block({best_config_naive[0][0]}, {best_config_naive[0][1]}): {best_config_naive[1]['naive']:.4f} ms")
print(f"  Shared Memory: Block({best_config_shared[0][0]}, {best_config_shared[0][1]}): {best_config_shared[1]['shared_memory']:.4f} ms")
