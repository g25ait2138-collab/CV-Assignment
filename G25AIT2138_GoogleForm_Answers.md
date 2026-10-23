# CV Assignment — Analysis QA (Google Form answers)
**Name:** Alok Sahni **Roll number:** G25AIT2138

---

## FINAL RESULTS
Format: Precision, Recall, mAP@0.5, mAP@0.5:0.95

**Result 1 — Original / Baseline Metrics**
```
0.6633, 0.5894, 0.6706, 0.5026
```

**Result 2 — Gaussian Blur Metrics**
```
0.6802, 0.5358, 0.6176, 0.4434
```

**Result 3 — Sharpened Metrics**
```
0.6271, 0.5616, 0.6245, 0.4550
```

---

## PREDICTION EXAMPLES

**Write 3 prediction examples from Original / Baseline Stage**
```
1. 000000000196.jpg (cluttered kitchen counter) — 29 detections, mean confidence 0.4761: 18 bowl, 6 carrot, 3 spoon, 1 dining table, 1 refrigerator. All small, densely packed items are resolved individually.
2. 000000000164.jpg (kitchen with glassware) — 11 detections, mean confidence 0.4059: 5 wine glass, 2 bottle, 1 cup, 1 chair, 1 microwave, 1 refrigerator. Thin stems and rims are visible, so wine glasses are correctly separated from bottles.
3. 000000000315.jpg (crowd under umbrellas) — 18 detections, mean confidence 0.4678: 10 umbrella, 6 person, 2 chair. Includes small, partially occluded people and umbrellas at the back of the scene.
```

**Write 3 prediction examples from Gaussian Blur Stage**
```
1. 000000000196.jpg — 19 detections (down from 29), mean confidence 0.5270: bowls fall 18 -> 10 and carrots 6 -> 2. Small adjacent objects merge once edges are smoothed, and refrigerator is now double-boxed (2).
2. 000000000164.jpg — 9 detections (down from 11), mean confidence 0.4664: ALL 5 wine glasses and the cup disappear, while bottle rises 2 -> 4. A class-confusion failure: the high-frequency stem and rim that separate a wine glass from a bottle are destroyed by the 9x9 kernel.
3. 000000000315.jpg — 15 detections (down from 18), mean confidence 0.4586: person 6 -> 5, umbrella 10 -> 9, chair 2 -> 1. The objects lost are the smallest and most occluded ones.
```

**Write 3 prediction examples from Sharpened Stage**
```
1. 000000000196.jpg — 19 detections, mean confidence 0.5196: partial recovery only — bowls 10 -> 11 and carrots 2 -> 3 versus the blurred stage, but still far below the 29 of the baseline. Refrigerator remains double-boxed (2).
2. 000000000164.jpg — 10 detections, mean confidence 0.4350: the wine glasses are STILL not recovered (0 wine glass, 4 bottle), and refrigerator is now double-boxed (2) where blur gave 1 — evidence that unsharp masking adds false positives on amplified texture but cannot restore destroyed detail.
3. 000000000315.jpg — 14 detections, mean confidence rose to 0.4768 (highest of the three stages for this image), but the count fell further (umbrella 9 -> 8). Sharpening raised confidence on surviving objects while edge haloes cost one borderline umbrella.
```

---

## ANALYSIS

**A. Effect of blur — 3 marks**
```
1. Localisation quality collapsed the most. mAP@0.5:0.95 fell 0.5026 -> 0.4434 (-0.0592, an 11.8% relative drop) while mAP@0.5 fell 0.6706 -> 0.6176 (-0.0530). The loss is larger at strict IoU thresholds than at IoU 0.50, so blur does not only remove objects, it also makes the surviving boxes less tightly aligned, because the edges that define the box extents are smeared.

2. Recall fell sharply, 0.5894 -> 0.5358 (-0.0536). The 9x9 Gaussian kernel is a low-pass filter; it removes exactly the high-frequency edge and texture evidence that the detector's early convolutional layers respond to, so small, weak and low-contrast objects stop firing above the confidence threshold.

3. Precision actually ROSE slightly, 0.6633 -> 0.6802 (+0.0169). This is not an improvement in the detector. Blur suppresses the marginal low-confidence detections first, so what survives is the small subset of large, high-contrast objects the model is still sure about. Fewer risky predictions means higher precision — the classic precision/recall trade-off, and the reason mAP (which accounts for both) still fell.

4. Example evidence: on 000000000196.jpg detections dropped 29 -> 19 (bowls 18 -> 10, carrots 6 -> 2). On 000000000164.jpg all 5 wine glasses and the cup vanished and bottle rose 2 -> 4, a class-confusion failure because the thin stem and rim are high-frequency structures. On 000000000315.jpg detections fell 18 -> 15, losing the small, partially occluded objects at the back.
```

**B. Why sharpening? — 2 marks**
```
1. Method used: unsharp masking, implemented directly as sharp = (1 + a) * I - a * G(I), with a = 1.5 and G a 9x9 Gaussian low-pass, clipped to [0, 255] with the image size unchanged.

2. Why it can help: a Gaussian blur is a low-pass filter, so the damage is specifically a loss of high-frequency content. The difference I - G(I) isolates precisely that missing high-frequency band, and adding a scaled copy of it back amplifies the attenuated edges. Local contrast at object boundaries is restored, and that boundary contrast is the evidence the detector's feature extractor depends on, so detections that fell just below the confidence threshold can cross back above it.

3. Why it cannot fully help: blurring is lossy and non-invertible. Unsharp masking only re-amplifies the frequencies that survived attenuation; it cannot recreate detail driven to zero. It is enhancement, not deblurring — true restoration needs deconvolution (e.g. Wiener) or a learned deblurring model. The amplification is also indiscriminate: sensor noise and JPEG artefacts are high-frequency too and get boosted with the real edges, and over-shoot haloes appear at strong edges.
```

**C. Did sharpening help? — 4 marks**
```
WHAT IMPROVED
1. Recall recovered, 0.5358 -> 0.5616 (+0.0258): restored edge contrast pushed borderline objects back over the detection threshold, the directly intended effect.
2. Both mAP figures improved: mAP@0.5 0.6176 -> 0.6245 (+0.0069) and mAP@0.5:0.95 0.4434 -> 0.4550 (+0.0116). Sharper boundaries give better-localised boxes, which is why the strict-IoU metric gained more than mAP@0.5.
3. Example evidence: on 000000000196.jpg bowls rose 10 -> 11 and carrots 2 -> 3; on 000000000315.jpg mean confidence rose 0.4586 -> 0.4768.

WHAT DID NOT IMPROVE
4. The recovery is small: only 19.6% of the mAP@0.5:0.95 lost to blur was regained (0.0116 of 0.0592) and only 13.0% of the mAP@0.5 loss. The sharpened stage is still well below baseline on every metric — mAP@0.5:0.95 0.4550 vs 0.5026, still -9.5% relative — and recall did not return to the 0.5894 baseline.
5. The wine-glass failure on 000000000164.jpg was not repaired at all: still 0 wine glass and 4 bottle. Sharpening amplifies surviving structure but cannot resurrect a stem the blur erased — the clearest proof that unsharp masking is enhancement, not deblurring.

WHAT BECAME WORSE
6. Precision DROPPED 0.6802 -> 0.6271 (-0.0531) and is now below even the original baseline of 0.6633 (-0.0362). This is the real cost: unsharp masking boosts noise, texture and compression artefacts along with true edges, so the detector fires on spurious structure and produces more false positives. On 000000000164.jpg refrigerator becomes double-boxed (1 -> 2) relative to the blurred stage.
7. On 000000000315.jpg the detection count fell further, 15 -> 14 (umbrella 9 -> 8), even though mean confidence rose — sharpening redistributed confidence rather than uniformly adding detections, and edge haloes cost a borderline case.
8. Net: the recall gain (+0.0258) was bought with a larger precision loss (-0.0531). mAP still improved overall only because better localisation offset that, and only marginally.
```

**D. Final conclusion — 1 mark (max 100 words)**
```
Yes, but only narrowly and with the trade-off stated. Unsharp masking recovered about a fifth of the mAP@0.5:0.95 lost to blur (0.4434 -> 0.4550) and clearly improved recall (0.5358 -> 0.5616), at zero training cost and negligible compute. But it cut precision to 0.6271, below even the original baseline of 0.6633, so it is worth keeping only where missed detections cost more than false alarms. It never approaches the 0.5026 baseline, so the real fix is to avoid blur at capture time, or to use a learned deblurring model instead of a fixed high-pass filter.
```
(96 words)
