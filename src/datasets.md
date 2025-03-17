# Biomedical datasets

There are several public data-sets for meshing:

(mri2fem2-chapter2)=

## MRI to FEM Vol 2 - Chapter 2

The following dataset: [DOI: 10.5281/zenodo.14536217](https://doi.org/10.5281/zenodo.14536217) contains the following surfaces (`stl`-files) for the patient known as Gonzo,
(under `Gonzo/outputs`) for a subset of regions in the brain.
Whenever there are files named `a.stl`, `final-a.stl`, it means that the initial surface `a.stl` has been processed, for instanced smoothened or perturbed for better surface quality with the resulting surface `final-a.stl`.

- Brainstem (`brainstem.stl`): The surface describing the brainstem
- Cerebellum (`cerebellum.stl`): The part of the brain known as "little brain"
- The dura (`dura.stl`, `final-dura.stl`): The outer layer towards the skull.
- The pial surfaces (`lhpial.stl`, `rhpial.stl`,`final-lhpial.stl`, `final-rhpial.stl`): The pial (outer) surface of the brain, separating grey-matter.
- The ventricles (`ventricles.stl`, `final-ventricles.stl`): A network of fluid cavities within the brain parenchyma.
- The white matter (`lhwhite.stl`, `rhwhite.stl`): The part of the brain parenchyma that is categorized as white matter. It lies within the grey matter.

The file `final-fm.stl`is a disk used to cut the brainstem.
The file `final-white.stl` contains the surface describing the white matter, the cerebellum and the brain-stem as one surface file.

The data can be extracted with:

```bash
wget -nc https://zenodo.org/records/14536218/files/mri2femii-chp2-dataset.tar.gz
tar xvzf mri2femii-chp2-dataset.tar.gz
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

(mri2fem2-chapter3)=

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
wget -nc https://zenodo.org/records/10808334/files/mhornkjol/mri2fem-ii-chapter-3-code-v1.0.0.zip
unzip mri2fem-ii-chapter-3-code-v1.0.0.zip
```

An example of a config file for `wildmeshing` is:

```json
{
  "operation": "union",
  "left": "dura.final.stl",
  "right": {
    "operation": "union",
    "left": "white.final.stl",
    "right": {
      "operation": "union",
      "left": "ventricles.final.stl",
      "right": {
        "operation": "union",
        "left": "rhchoroid.plexus.final.stl",
        "right": "lhchoroid.plexus.final.stl"
      }
    }
  }
}
```

(data-from-the-paper-in-silico-molecular-enrichment-and-clearance-of-the-human-intracranial-space)=

## Data from the paper In-silico molecular enrichment and clearance of the human intracranial space

Data [DOI: 10.5281/zenodo.14749162](https://doi.org/10.5281/zenodo.14749162) from the paper by Marius Causemann et al. 2025 [DOI: 10.1101/2025.01.30.635680 ](https://doi.org/10.1101/2025.01.30.635680).

There following regions of the brain has been marked:

- The dura (`skull.ply`): The outer layer towards the skull
- The brain parenchyma (`parenchyma_incl_ventr.ply`): The part of the brain containing white and grey matter, both the Cerebrum and the Cerebellum.
- The lateral ventricles (`LV.ply`): The part of the ventricular network where cerebrospinal fluid is produced.
- The third and fourth ventricles (`V34.ply`): The part of the ventricular network that connects the lateral ventricles to the subarachnoid space.

Other surfaces not used directly in the mesh generation is `parenchyma.ply`, which includes an interface towards the ventricles. The surface `cerebrum.ply` is also not used, as it does not contain the cerebellum (little brain).

The surfaces can be extracted with:

```bash
wget -nc https://zenodo.org/records/14749163/files/surfaces.zip
unzip surfaces.zip
```

An example configuration for wildmeshing would be

```json
{
  "operation": "union",
  "left": "skull.ply",
  "right": {
    "operation": "union",
    "left": "parenchyma_incl_ventr.ply",
    "right": {
      "operation": "union",
      "left": "LV.ply",
      "right": "V34.ply"
    }
  }
}
```
