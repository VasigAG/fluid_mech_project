# Mechanical Biomarker Extraction from Pathological Red Blood Cells

Computational biomechanics study of pathological red blood cells (RBCs) using the HEMOCELL framework, focused on extracting mechanically meaningful biomarkers relevant to cross-slot microfluidic diagnostics and stagnation-point extensional flow experiments.

---

## Overview

This project simulates deformable RBC transport in microvascular-scale flow using:

- **HEMOCELL v2.x**
- **Lattice Boltzmann Method (LBM)**
- **Immersed Boundary Method (IBM)**

The goal is to establish biomechanical phenotype signatures for different haematological conditions and identify features that could be measured experimentally in a cross-slot microfluidic device using optical imaging.

The simulated disease states are:

- Healthy RBCs
- Sickle Cell Disease
- Thalassaemia α
- Thalassaemia β
- Malaria (*P. falciparum*)
- Diabetes Mellitus

---

# Scientific Motivation

Mechanical properties of RBCs encode clinically important information:

| Property | Diagnostic relevance |
|---|---|
| Membrane deformability | Sickle cell rigidity, malaria infection |
| Volume variability | Osmotic instability |
| Lateral migration | Cell-free layer formation |
| Tortuosity | Flow disruption |
| Speed fluctuations | Near-wall rheological instability |
| Surface-area-to-volume ratio | Microcytic morphology |

The project aims to determine whether these features can be extracted non-invasively using extensional microfluidic flow.

---
