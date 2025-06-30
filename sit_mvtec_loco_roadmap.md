# Roadmap: Adapting **SiT‑Small** to MVTec LOCO AD

> Plain‑language guide that explains **what** each step achieves.  
> Use it as a compass while you fill in code details in the SiT repository.

---

## 1. Clarify the End Goal
- **Task**: Teach the model what *normal* looks like for all five LOCO product families so it can flag any deviation—scratches, missing parts, wrong labels, etc.  
- **Constraint**: Only defect‑free images are shown during training.

---

## 2. Re‑use What SiT Already Knows
- The ImageNet‑pre‑trained weights already encode generic shapes, edges and colours.  
- Fine‑tuning will **specialise** these representations to the exact layouts and textures of LOCO products.

---

## 3. Prepare the LOCO Data
1. **Sort images** into:  
   - `train_good` – all normal images  
   - `validation_good` – small normal subset for threshold tuning  
   - `test` – mixture of good and anomalous  
2. Keep the original resolution; SiT handles patchification internally.  
3. Ground‑truth **masks** are only for evaluation, never for training.

---

## 4. Choose the Learning Objective
| Objective inside the network | Why it matters for LOCO |
|------------------------------|-------------------------|
| **Reconstruct hidden blocks** | Forces understanding of correct component arrangement; a missing or wrong part becomes hard to fill in and yields high error. |
| *(Optional)* **Light global regulariser** | A tiny dose of SiT’s contrastive signal can increase robustness to lighting/viewpoint changes **without** fragmenting the normal manifold. |

> In practice, keep reconstruction dominant; make any contrastive weight *very small*.

---

## 5. Outline the Fine‑tuning Process
1. **Allow all layers to adapt** – do not freeze the encoder.  
2. **Mask strategy** – hide ≈ 65 % of the image in contiguous rectangles; this teaches long‑range context.  
3. **Training loop goal** – minimise pixel difference on hidden blocks so the network perfects the notion of “normal”.  
4. **Validation check** – confirm reconstruction error on `validation_good` images keeps decreasing.

---

## 6. Plan the Scoring Logic
- **Image‑level flag**: average the reconstruction error; scores above threshold → anomaly.  
- **Pixel‑level heat‑map**: retain per‑patch errors for visual inspection and sPRO metric.  
- **Threshold**: pick from the 99.5 th percentile of validation normals.

---

## 7. Evaluate with LOCO Metrics
| Metric | What it tells you |
|--------|------------------|
| **Image‑level AUROC** | Separation of good vs defective shots. |
| **Pixel‑level sPRO** | Overlap between heat‑map and true defect regions. |
| **Structural vs Logical splits** | Performance on low‑level scratches vs high‑level part errors. |

---

## 8. Iterate If Needed
- **Logical misses** → use larger or multiple masks.  
- **Too many false alarms** → reduce or remove contrastive component, or lower mask ratio.  
- **Blurry heat‑maps** → explore reconstructing discrete colour tokens instead of raw RGB.

---

## 9. Reproducibility Checklist
- Record random seed, mask ratio, and any loss weightings.  
- Note environment details (Python & library versions).

---

## 10. What Success Looks Like
- **≥ 0.90 AUROC** on both structural *and* logical subsets.  
- **Sharp heat‑maps** that highlight only the faulty region.  
- **Stable threshold** that generalises across categories.

---

*Use this roadmap to guide your adaptations of the SiT repository; fill in commands, paths and parameter flags as you implement each step.*
