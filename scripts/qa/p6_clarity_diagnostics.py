"""Summarize a completed P6 run without tuning its candidates.

Run: .venv/bin/python scripts/qa/p6_clarity_diagnostics.py RUN_DIRECTORY
Plots are diagnostics, not perceptual ground truth. PNG crops remain native scale.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='p6-mpl-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def number(row, field):
    return float(row[field]) if row[field] else None


def run(root):
    manifest = json.loads((root/'manifest.json').read_text())
    rows = read(root/'synthetic.csv')
    grouped = {}
    for row in rows:
        key = (row['fixture'], row['arm'], row['recipe_clarity'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (fixture, arm, value), group in grouped.items():
        result = dict(fixture=fixture, arm=arm, recipe_clarity=int(value))
        for metric in ('paired_noise_rms_DN', 'highpass_rms_DN', 'haar_MAD_DN', 'PSD_high_DN2',
                       'block_excess_DN2', 'clean_structure_gain', 'overshoot_DN', 'undershoot_DN',
                       'halo_area_DN_pixels', 'changed_u8_pixel_fraction', 'quantized_change_rms_DN',
                       'bypass_fraction'):
            values = [number(r, metric) for r in group if r[metric]]
            result[metric] = float(np.mean(values)) if values else None
        for metric in ('paired_noise_rms_DN', 'highpass_rms_DN', 'haar_MAD_DN', 'PSD_high_DN2'):
            baseline = grouped[(fixture, 'disabled', value)]
            ratios = [float(a[metric])/float(b[metric]) for a, b in zip(group, baseline) if float(b[metric]) > 1e-8]
            result[metric+'_ratio'] = float(np.mean(ratios)) if ratios else None
            result[metric+'_ratio_minmax'] = [float(min(ratios)), float(max(ratios))] if ratios else None
        summary.append(result)
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    display = ('disabled', 'production_current', 'matched_current', 'soft_floor', 'variance_snr',
               'multiscale_snr', 'snr_edge_cap', 'edge_coherence')
    colors = dict(zip(display, ('black', '#d62728', '#ff7f0e', '#9467bd', '#2ca02c', '#1f77b4', '#8c564b', '#17becf')))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, fixture in zip(axs.flat, ('gaussian_2DN', 'correlated_noise', 'jpeg_gradient_q40', 'quantized_gradient')):
        for arm in display:
            selected = sorted([r for r in summary if r['fixture'] == fixture and r['arm'] == arm], key=lambda r:r['recipe_clarity'])
            ax.plot([r['recipe_clarity'] for r in selected], [r['paired_noise_rms_DN_ratio'] for r in selected], label=arm, color=colors[arm])
        ax.set_title(fixture)
        ax.set_xlabel('Recipe clarity (strength = value / 100)')
        ax.set_ylabel('Paired corruption RMS / disabled')
        ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('P6: corruption response, 3 fixed seeds (not image-quality scores)')
    fig.tight_layout()
    fig.savefig(root/'noise_sweep.png', dpi=160)
    plt.close(fig)

    profiles = read(root/'edge_profiles.csv')
    halo_rows = []
    for fixture in ('hard_edge', 'blurred_edge', 'jpeg_blurred_edge'):
        for value in (4, 10):
            for arm in manifest['arms']:
                data = [r for r in profiles if r['fixture'] == fixture and r['arm'] == arm and int(r['recipe_clarity']) == value]
                profile = np.array([float(r['output_DN']) for r in data])
                clean = np.array([float(r['clean_DN']) for r in data])
                n = len(profile)
                lo, hi = np.median(profile[12:n//4]), np.median(profile[3*n//4:-12])
                # Remove far-plateau conversion offsets before calling a delta a halo.
                offsets = np.where(np.arange(n) < n//2, lo-np.median(clean[12:n//4]), hi-np.median(clean[3*n//4:-12]))
                corrected = profile-clean-offsets
                beyond = (abs(np.arange(n)-n//2) > 6) & (np.arange(n) >= 12) & (np.arange(n) < n-12)
                halo_rows.append(dict(fixture=fixture, recipe_clarity=value, arm=arm,
                    plateau_relative_undershoot_DN=float(max(0, lo-profile.min())),
                    plateau_relative_overshoot_DN=float(max(0, profile.max()-hi)),
                    halo_area_beyond6_DN_pixels=float(abs(corrected[beyond]).sum()),
                    halo_peak_beyond6_DN=float(abs(corrected[beyond]).max())))
    with (root/'edge_halo_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(halo_rows[0]))
        writer.writeheader()
        writer.writerows(halo_rows)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, fixture in enumerate(('hard_edge', 'blurred_edge', 'jpeg_blurred_edge')):
        for row, value in enumerate((4, 10)):
            ax = axs[row, col]
            for arm in display:
                data = [r for r in profiles if r['fixture'] == fixture and r['arm'] == arm and int(r['recipe_clarity']) == value]
                x = np.array([float(r['x']) for r in data])-192
                delta = [float(r['output_DN'])-float(r['clean_DN']) for r in data]
                ax.plot(x, delta, label=arm, color=colors[arm])
            ax.set_xlim(-25, 25)
            ax.set_title(f'{fixture}, recipe {value}')
            ax.set_xlabel('Pixels from transition')
            ax.set_ylabel('Output minus known clean (DN)')
            ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('P6: signed native edge profiles (midrange, not clipping-hidden)')
    fig.tight_layout()
    fig.savefig(root/'halo_profiles.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    fixtures = ('hair_lines', 'fabric', 'pores', 'printed')
    for i, arm in enumerate(display[1:]):
        gains = [next(r for r in summary if r['fixture']==f and r['arm']==arm and r['recipe_clarity']==10)['clean_structure_gain'] for f in fixtures]
        ax.bar(np.arange(len(fixtures))+(i-3)*.105, np.array(gains)-1, width=.10, label=arm, color=colors[arm])
    ax.set_xticks(np.arange(len(fixtures)), fixtures)
    ax.set_ylabel('Clean structure projection gain minus 1')
    ax.set_title('Useful synthetic detail boost at recipe 10; not pore authenticity evidence')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(root/'structure_gain.png', dpi=160)
    plt.close(fig)

    if (root/'mixed_detail.csv').exists():
        mixed = read(root/'mixed_detail.csv')
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, arm in enumerate(display[1:]):
            gains = [np.mean([float(r['structure_in_noise_projection_gain']) for r in mixed
                             if r['fixture']==f+'_noisy' and r['arm']==arm and r['recipe_clarity']=='10']) for f in fixtures]
            ax.bar(np.arange(len(fixtures))+(i-3)*.105, np.array(gains)-1, width=.10, label=arm, color=colors[arm])
        ax.set_xticks(np.arange(len(fixtures)), fixtures)
        ax.set_ylabel('Matched-noise structural projection gain minus 1')
        ax.set_title('Structure IN 2-DN noise, recipe 10; flat + identical noise counterfactual')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(root/'mixed_structure_gain.png', dpi=160)
        plt.close(fig)

    if (root/'real_edge_profiles.csv').exists():
        data = read(root/'real_edge_profiles.csv')
        boundaries = list(dict.fromkeys(r['roi'] for r in data))
        fig, axs = plt.subplots(len(boundaries), 1, figsize=(13, 3*len(boundaries)), squeeze=False)
        for ax, roi in zip(axs.flat, boundaries):
            for arm in ('production_current', 'variance_snr', 'multiscale_snr', 'snr_edge_cap'):
                selected = [r for r in data if r['roi'] == roi and r['arm'] == arm and r['recipe_clarity'] == '4' and r['channel'] == 'G']
                ax.plot([float(r['x']) for r in selected], [float(r['signed_delta_DN']) for r in selected], label=arm, color=colors[arm])
            ax.set_title(roi+' — 5-row native profile, green channel, recipe 4')
            ax.set_xlabel('Native source x')
            ax.set_ylabel('Output minus source DN')
            ax.grid(alpha=.2)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle('Real boundaries: signed changes, NOT clean-reference halo estimates')
        fig.tight_layout()
        fig.savefig(root/'real_boundary_deltas.png', dpi=160)
        plt.close(fig)

    parts = ['<!doctype html><meta charset="utf-8"><title>P6 native diagnostic review</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#eee} .strip{display:flex;overflow:auto;gap:12px}figure{margin:0}img.native{max-width:none}figcaption{position:sticky;left:0} h2{margin-top:40px}</style>',
             '<h1>P6 offline review</h1><p>Research only. One rendered JPEG, no clean sensor reference. '
             'Use browser zoom 100%: native images have no CSS resizing. Horizontal strips scroll; '
             'open a PNG separately for pixel inspection. The 8× detail map is exaggerated diagnostic data, not a delivered photo.</p>',
             '<p><a href="manifest.json">Manifest / coordinates / hashes</a> · <a href="synthetic.csv">Synthetic metrics</a> · '
             '<a href="parameter_sensitivity.csv">Parameter sweep</a> · <a href="real_rois.csv">Real ROI metrics</a> · '
             '<a href="edge_profiles.csv">Signed profiles</a></p>']
    for plot in ('noise_sweep.png', 'halo_profiles.png', 'structure_gain.png', 'mixed_structure_gain.png', 'real_boundary_deltas.png'):
        if not (root/plot).exists():
            continue
        parts.append(f'<p><a href="{plot}"><img src="{plot}" width="1000" alt="{plot}"></a></p>')
    for roi in manifest['roi_xyxy']:
        parts.append(f'<h2>{html.escape(roi)} — {manifest["roi_xyxy"][roi]}</h2>')
        for value in (4, 10):
            parts.append(f'<h3>Recipe {value}</h3><div class="strip">')
            for arm in ('input', 'production_current', 'matched_current', 'soft_floor', 'variance_snr', 'multiscale_snr', 'snr_edge_cap', 'edge_coherence', 'detail_8x'):
                name = f'DSCF1884_{roi}_{arm}.png' if arm in ('input', 'detail_8x') else f'DSCF1884_{roi}_{arm}_{value:03d}.png'
                parts.append(f'<figure><figcaption>{arm}</figcaption><a href="{name}"><img class="native" src="{name}" alt="{roi} {arm}"></a></figure>')
            parts.append('</div>')
    (root/'review.html').write_text('\n'.join(parts)+'\n')
    print(root/'review.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    run(parser.parse_args().run_directory)
