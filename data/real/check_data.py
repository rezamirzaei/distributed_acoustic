#!/usr/bin/env python3
"""Check downloaded data files"""
import numpy as np
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("CHECKING DOWNLOADED REAL DATA FILES")
print("=" * 60)

# Check Ridgecrest data
ridgecrest_file = os.path.join(data_dir, 'ridgecrest_m71_das_array.npz')
if os.path.exists(ridgecrest_file):
    d = np.load(ridgecrest_file, allow_pickle=True)
    print(f"\n1. Ridgecrest M7.1 Earthquake Data:")
    print(f"   File size: {os.path.getsize(ridgecrest_file) / 1e6:.2f} MB")
    print(f"   Data shape: {d['data'].shape}")
    print(f"   Duration: {d['time'][-1]:.1f} seconds")
    print(f"   Sampling rate: {d['sampling_rate']} Hz")
    print(f"   Event: {d['event']}")
    print(f"   Source: {d['source']}")
else:
    print("Ridgecrest data not found!")

# Check catalog
catalog_file = os.path.join(data_dir, 'ridgecrest_earthquake_catalog.csv')
if os.path.exists(catalog_file):
    with open(catalog_file) as f:
        lines = f.readlines()
    print(f"\n2. USGS Earthquake Catalog:")
    print(f"   File size: {os.path.getsize(catalog_file) / 1e6:.2f} MB")
    print(f"   Events: {len(lines) - 1}")
else:
    print("Catalog not found!")

print("\n" + "=" * 60)
