import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Global Memory Bandwidth (test_gmem)
# ============================================================
gmem_strides = [1, 2, 4, 8]
gmem_bw = [530.113, 182.494, 92.001, 46.287]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar([str(s) for s in gmem_strides], gmem_bw, color='#4C72B0', width=0.5, edgecolor='black', linewidth=0.5)
ax.set_xlabel('STRIDE', fontsize=13)
ax.set_ylabel('Bandwidth (GB/s)', fontsize=13)
ax.set_title('Global Memory Bandwidth vs STRIDE', fontsize=14)
ax.set_ylim(0, max(gmem_bw) * 1.15)
for bar, val in zip(bars, gmem_bw):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('gmem_bandwidth.png', dpi=150)
plt.close()
print('Saved gmem_bandwidth.png')

# ============================================================
# 2. Shared Memory Bandwidth (test_smem) — heatmap
# ============================================================
strides = [1, 2, 4, 8, 16, 32]
bitwidths = [2, 4, 8]
smem_data = np.array([
    [4306.19, 4232.70, 2157.94, 830.395, 426.971, 215.573],
    [8612.06, 4323.87, 2025.75, 1017.59, 509.125, 251.539],
    [8643.58, 4339.48, 2173.55, 1087.67, 544.072, 544.071],
])

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(smem_data, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(strides)))
ax.set_xticklabels([str(s) for s in strides])
ax.set_yticks(range(len(bitwidths)))
ax.set_yticklabels([str(bw) for bws in bitwidths for bw in [bws]])
ax.set_xlabel('STRIDE', fontsize=13)
ax.set_ylabel('BITWIDTH', fontsize=13)
ax.set_title('Shared Memory Bandwidth (GB/s)', fontsize=14)

# Annotate each cell with the value
for i in range(len(bitwidths)):
    for j in range(len(strides)):
        val = smem_data[i, j]
        color = 'white' if val > 4000 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Bandwidth (GB/s)', fontsize=12)
plt.tight_layout()
plt.savefig('smem_bandwidth.png', dpi=150)
plt.close()
print('Saved smem_bandwidth.png')
