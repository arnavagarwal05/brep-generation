# Data format

Training reads `data/processed_dataset/{train,validation,test}/<sub>/<id>_solidgen.json`,
with the split listed in `data/train_val_test_split.json` as `{"train": ["<sub>/<id>", ...], ...}`.

Each `_solidgen.json` is an indexed B-rep:

```json
{
  "vertices": [[x, y, z], ...],          // float, model-space coordinates
  "edges":    [[v0, v1], [v0, v1, v2]],  // vertex indices; 2 = line, 3 = circular arc through a midpoint
  "faces":    [[e0, e1, e2, e3], ...]    // edge indices forming a closed loop
}
```

`sample/00000070_solidgen.json` is one real example (32 vertices, 40 edges, 18 faces).

The files were produced from the DeepCAD JSON models by a curation script that
rebuilt each model's B-rep, filtered to 8 to 130 faces, quantised vertices to a
64-level grid, and dropped models where quantisation merged vertices or produced
duplicate geometry. That script ran on the Arizona State University cluster during
the internship and was not preserved; `logs/` has no copy of its output either.
Anything that emits the schema above will work.
