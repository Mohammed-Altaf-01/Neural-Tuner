## Episode Trace: `inception_v3` (medium)

### Random Agent (no profiling)
**Step 1:** `quantize_layer(conv_stem, FP32)` → _QUANTIZE: conv_stem  FP32 → FP32
  WARNING: Layer not profiled. Sensitivity unkn_
**Step 2:** `quantize_layer(conv_bn_1, FP32)` → _QUANTIZE: conv_bn_1  FP32 → FP32
  WARNING: Layer not profiled. Sensitivity unkn_
**Step 3:** `quantize_layer(mixed_3a, INT8)` → _QUANTIZE: mixed_3a  FP32 → INT8
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 4:** `quantize_layer(mixed_4a, FP16)` → _QUANTIZE: mixed_4a  FP32 → FP16
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 5:** `quantize_layer(mixed_5a, FP16)` → _QUANTIZE: mixed_5a  FP32 → FP16
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 6:** `quantize_layer(mixed_6a, FP16)` → _QUANTIZE: mixed_6a  FP32 → FP16
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 7:** `quantize_layer(mixed_7a, FP32)` → _QUANTIZE: mixed_7a  FP32 → FP32
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 8:** `quantize_layer(avg_pool, FP32)` → _QUANTIZE: avg_pool  FP32 → FP32
  WARNING: Layer not profiled. Sensitivity unkno_
**Step 9:** `quantize_layer(dropout, INT4)` → _QUANTIZE: dropout  FP32 → INT4
  WARNING: Layer not profiled. Sensitivity unknow_
**Step 10:** `quantize_layer(fc_classifier, FP32)` → _QUANTIZE: fc_classifier  FP32 → FP32
  WARNING: Layer not profiled. Sensitivity _
**Step 11:** `benchmark()`
**Step 12:** `submit()`

**Final reward: 0.3037** | constraints_met=False

### Heuristic Agent (profile-first)
**Step 1:** `profile_layer(conv_stem)` → _sensitivity=0.040_
**Step 2:** `profile_layer(conv_bn_1)` → _sensitivity=0.020_
**Step 3:** `profile_layer(mixed_3a)` → _sensitivity=0.080_
**Step 4:** `profile_layer(mixed_4a)` → _sensitivity=0.120_
**Step 5:** `profile_layer(mixed_5a)` → _sensitivity=0.090_
**Step 6:** `profile_layer(mixed_6a)` → _sensitivity=0.150_
**Step 7:** `quantize_layer(conv_stem, INT4)`
**Step 8:** `quantize_layer(conv_bn_1, INT4)`
**Step 9:** `quantize_layer(mixed_3a, INT4)`
**Step 10:** `quantize_layer(mixed_4a, INT8)`
**Step 11:** `quantize_layer(mixed_5a, INT4)`
**Step 12:** `quantize_layer(mixed_6a, INT8)`
**Step 13:** `benchmark()`
**Step 14:** `submit()`

**Final reward: 0.6428** | constraints_met=False
