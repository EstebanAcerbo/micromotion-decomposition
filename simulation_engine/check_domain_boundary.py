"""
check_domain_boundary.py
========================
Visualises the simulation domain, the fixed cell culture (shown as the
central cell only, for clarity), and the electrode circle for each
electrode offset dx in [0, 5, 10, 15, 20, 25, 30, 35, 40] with dy=0.

Geometry note
-------------
In Library A (electrode offset sweep), the CELLS ARE FIXED and the
ELECTRODE CENTRE moves.  substrate_offset() shifts the electrode by
(x_px, y_px) pixels relative to the domain centre.  The culture
rasterisation is independent of the electrode position.

For each subplot:
  - Grey rectangle  : the full simulation domain (side × side pixels)
  - Blue filled rect: the central cell (spacing_x × spacing_y pixels
                      centred at the domain centre)
  - Orange circle   : the electrode at the offset position
  - Red outline     : drawn instead of orange when any part of the
                      electrode comes within 2 px of a domain boundary

Output: results/check_domain_boundary.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

from engine import build_geometry

RESULTS = os.path.join(_HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

NEAR_THRESHOLD = 2   # pixels


def main():
    print('Building confluent geometry ...')
    geo = build_geometry()

    side  = geo['side']          # domain side length [px]
    sx    = geo['stencil'].spacing_x   # cell spacing x [px]  (= width_px  for RecCell)
    sy    = geo['stencil'].spacing_y   # cell spacing y [px]  (= height_px for RecCell)
    de    = geo['p']['de']        # electrode diameter [cm]
    d     = geo['d']              # pixel size [cm]

    # Electrode radius in pixels
    elec_r_px = (de / 2.0) / d

    # Domain centre in pixel coordinates
    dom_cx = side / 2.0   # = side * d / 2  / d
    dom_cy = side / 2.0

    # Central cell bounds (cell centre is at domain centre for the base culture)
    # The stencil is centred on the cell centre; TJ is the outermost ring.
    # For visualisation we draw the full stencil bounding box (sx × sy).
    cell_cx = dom_cx   # pixel coords of central cell centre
    cell_cy = dom_cy

    dx_values = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    dy = 0

    print(f'Domain side       : {side} px')
    print(f'Cell spacing x    : {sx} px')
    print(f'Cell spacing y    : {sy} px')
    print(f'Electrode radius  : {elec_r_px:.2f} px  (diameter {2*elec_r_px:.2f} px)')
    print(f'Pixel size d      : {d*1e4:.4f} µm')
    print()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes_flat = axes.ravel()

    for i, dx in enumerate(dx_values):
        ax = axes_flat[i]

        # Electrode centre in pixel coords
        elec_cx = dom_cx + dx
        elec_cy = dom_cy + dy

        # Distance from electrode edge to each domain boundary [px]
        dist_left   = elec_cx - elec_r_px          # to left edge  (x=0)
        dist_right  = (side - 1) - (elec_cx + elec_r_px)   # to right edge (x=side-1)
        dist_bottom = elec_cy - elec_r_px          # to bottom edge (y=0)
        dist_top    = (side - 1) - (elec_cy + elec_r_px)   # to top edge   (y=side-1)

        min_margin = min(dist_left, dist_right, dist_bottom, dist_top)
        near_boundary = min_margin < NEAR_THRESHOLD

        print(f'dx={dx:3d} px | elec centre ({elec_cx:.1f}, {elec_cy:.1f}) px '
              f'| margin to boundary: '
              f'L={dist_left:.1f}  R={dist_right:.1f}  '
              f'B={dist_bottom:.1f}  T={dist_top:.1f} px'
              + ('  <<< NEAR BOUNDARY' if near_boundary else ''))

        # ── Draw domain ───────────────────────────────────────────────────────
        domain_rect = patches.Rectangle(
            (0, 0), side - 1, side - 1,
            linewidth=1.5, edgecolor='black', facecolor='#e8e8e8', zorder=1,
        )
        ax.add_patch(domain_rect)

        # ── Draw central cell (fixed, always at domain centre) ────────────────
        cell_rect = patches.Rectangle(
            (cell_cx - sx / 2, cell_cy - sy / 2), sx, sy,
            linewidth=1.2, edgecolor='steelblue', facecolor='steelblue',
            alpha=0.4, zorder=3, label='Central cell',
        )
        ax.add_patch(cell_rect)

        # ── Draw electrode circle ─────────────────────────────────────────────
        elec_color  = 'red'    if near_boundary else 'darkorange'
        elec_fill   = '#ffcccc' if near_boundary else '#ffe0b2'
        elec_lw     = 2.5      if near_boundary else 1.5

        elec_circle = patches.Circle(
            (elec_cx, elec_cy), elec_r_px,
            linewidth=elec_lw, edgecolor=elec_color, facecolor=elec_fill,
            alpha=0.6, zorder=4,
        )
        ax.add_patch(elec_circle)

        # Mark domain boundaries (bold lines)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.plot([0, side-1, side-1, 0, 0], [0, 0, side-1, side-1, 0],
                color='black', lw=2, zorder=5)

        # Mark electrode centre
        ax.plot(elec_cx, elec_cy, '+', color=elec_color, ms=8, mew=2, zorder=6)

        # ── Axes ──────────────────────────────────────────────────────────────
        pad = 20
        ax.set_xlim(-pad, side - 1 + pad)
        ax.set_ylim(-pad, side - 1 + pad)
        ax.set_aspect('equal')
        ax.set_xticks([0, side // 2, side - 1])
        ax.set_yticks([0, side // 2, side - 1])
        ax.tick_params(labelsize=7)

        # Title
        title = f'dx = {dx} px'
        if near_boundary:
            title += '\n** NEAR BOUNDARY **'
            ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        else:
            ax.set_title(title, fontsize=9, color='black')

        # Annotate minimum margin
        ax.annotate(
            f'min margin: {min_margin:.1f} px',
            xy=(0.03, 0.04), xycoords='axes fraction',
            fontsize=7.5, color=elec_color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    # ── Global labels ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'Electrode offset sweep  |  domain = {side}×{side} px'
        f'  |  elec diam = {2*elec_r_px:.1f} px'
        f'  |  d = {d*1e4:.3f} µm/px'
        f'\nCell (blue) fixed; electrode (orange/red) shifts by dx along x'
        f'  |  NEAR-BOUNDARY threshold = {NEAR_THRESHOLD} px',
        fontsize=10, y=0.98,
    )

    # Legend (shared)
    legend_handles = [
        patches.Patch(facecolor='#e8e8e8', edgecolor='black', label=f'Domain ({side}×{side} px)'),
        patches.Patch(facecolor='steelblue', alpha=0.5, label=f'Central cell ({sx}×{sy} px)'),
        patches.Patch(facecolor='#ffe0b2', edgecolor='darkorange', label=f'Electrode (r≈{elec_r_px:.0f} px), OK'),
        patches.Patch(facecolor='#ffcccc', edgecolor='red', label=f'Electrode, NEAR BOUNDARY (<{NEAR_THRESHOLD} px)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = os.path.join(RESULTS, 'check_domain_boundary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
