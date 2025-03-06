# Instructions for meshing of brains

There are several public data-sets for meshing

## MRI to FEM Vol 2 - Chapter 3

The following dataset: [DOI: 10.5281/zenodo.10808333](https://doi.org/10.5281/zenodo.10808333)
contains surfaces (`stl`-files) for a subset of regions in the brain:

- The dura (`dura.final.stl`): The outer layer towards the skull
- Cisterna Magna (`cisterna-magna.stl`): A fluid cavity between the cerebellum and the brain-stem
- The choroid plexus (`lhchoroid.plexus.final.stl` and `rhchoroid.plexus.final.stl`): A fluid cavity within the ventricles, where cells produce cerebrospinal fluid
- The ventricles (`ventricles.final.stl`): A network of fluid cavities within the brain parenchyma.
- The pial surface (`lhpial.final.stl`, `rhpial.final.stl`): The pial (outer) surface of the brain, separating grey-matter from cerebrospinal fluid in the sub-arachnoid spaces (the volume between the pial surface and the dura matter).
- The white matter (`white.final.stl`): The part of the brain parenchyma that is categorized as white matter. It lies within the grey matter.

The data can be extracted with:

```bash
wget -nc https://zenodo.org/records/10808334/files/mhornkjol/mri2fem-ii-chapter-3-code-v1.0.0.zip && \
unzip mri2fem-ii-chapter-3-code-v1.0.0.zip
```

An example of a config file for `wildmeshing` is:

```json
{
  "operation": "union",
  "left": "final-dura.stl",
  "right": {
    "operation": "union",
    "left": {
      "operation": "union",
      "left": "final-lhpial.stl",
      "right": "final-rhpial.stl"
    },
    "right": {
      "operation": "union",
      "left": "final-white.stl",
      "right": "final-ventricles.stl"
    }
  }
}
```
