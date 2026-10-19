# Radar data adapters

Loaders yield normalized [B,T,1,H,W] tensors. Training drops an incomplete microbatch; validation/test retain it. HDF5 handles are not shared across workers.

| Dataset | Supplied preprocessing | Evaluation scale | Thresholds |
|---|---|---|---|
| SEVIR VIL | Divide encoded values by 255; resize | 255 | 16,74,133,160,181,219 |
| MeteoNet | Divide values by 90; resize | 90 | 12,18,24,32 |
| Shanghai | Divide encoded pixels by 255; resize | 90 | 20,30,35,40 |
| CIKM | Divide encoded pixels by 255; center crop/pad | 80 | 20,30,35,40 |

The original resize/crop operations are preserved. CIKM's 101x101 input is zero-padded by center crop when requesting 128x128; it is not resized.

## HDF5 layout and split mappings

- Shanghai: train/test groups each contain scalar all_len and entries 0,1,... with shape [T,H,W]. Original validation maps to test.
- MeteoNet: top-level train_len/test_len scalars; train/test groups contain entries 0,1,... . Original validation maps to test.
- CIKM: top-level train_len/test_len scalars; groups contain sample_1,sample_2,... . Original validation uses up to the first 1000 test entries.
- SEVIR: root contains CATALOG.csv and data/ with event files referenced by the catalog. Default window length is 25 and stride 13. Original date filters are retained: training time <= 2019-01-01; validation 2019-01-01 < time <= 2019-06-01; test 2019-06-01 < time <= 2019-12-31.

The overlapping validation/test mappings in the first three adapters are retained for source compatibility and must be considered in experiment reporting. This refactor does not create independent validation splits.

get_dataset returns the original six-tuple with unrequested splits set to None. build_dataloaders returns DataLoaders in the first three positions. It reads one SEVIR window per Dataset item, then uses standard batching so distributed evaluation identifies real tail samples.
