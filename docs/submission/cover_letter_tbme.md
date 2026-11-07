# Cover letter: IEEE Transactions on Biomedical Engineering

Fill in the bracketed fields. Keep the letter to one page. A supervisor or
co-author named here strengthens the submission.

---

[Date]

The Editor-in-Chief
IEEE Transactions on Biomedical Engineering

Dear Professor [surname],

Please consider the enclosed manuscript, **"Screening and Auditing Data Leakage in
Wearable Physiological Regression, with Application to PPG-Based Glucose and
Blood Pressure,"** for publication in IEEE Transactions on
Biomedical Engineering as a regular paper.

**The problem.** Cuffless estimation of glucose and blood pressure from
photoplethysmography (PPG) is a major engineering target, and published
errors often approach those of regulated devices. Many of these results do
not survive a change of evaluation protocol. Subject-wise splitting, the
usual remedy, does not detect leaks created while a dataset is built or
augmented, and the size of a leak depends on the model as well as the
split.

**The contribution.** We present an evaluation method for this problem and
validate it as a method. Its most novel part is an *a-priori* screen for
identity-leak inflation: label clustering times how well the model's inputs
identify a subject, computable before any model is trained. We tested it
prospectively: predictions for five new model types were committed to the
public repository before the models were run. The screen flagged every
leaking input set correctly and gave the size of the inflation within a
factor of two in 60% of the 20 new model–cohort pairs. It placed the
blood-pressure leak in demographic inputs: a model given only age, sex,
height and weight, without any PPG, reaches MAE 5.9 mmHg and r = 0.93 under a
random split, and is no better than the mean once subjects are separated.

The full method:

1. **A leakage audit** with six parts, the screen plus: a detector for label-construction
   errors, including a block-scan variant for partial corruption; a copy
   probe that flags augmented or duplicated samples crossing the split;
   split protocols whose no-leak guarantees are asserted in code; a leakage
   ladder that measures inflation one mechanism at a time; and a
   null-anchored decision rule with subject-cluster bootstrap intervals and
   controls.
2. **Validation on known ground truth.** The detector found every corrupted
   subject for cycle lengths of three or more with no false alarms, on
   synthetic data and on 63 real subjects; its failure modes (shuffled or
   window-major tables) are measured and stated. Over 1,000 no-signal
   cohorts the decision rule's false-claim rate was 0.4–0.5% under honest
   splits and 16–99.9% under a leaky one.
3. **Biomedical significance shown on real problems.**
   - A widely used public glucose table carries a label-replication error
     that subject-wise splitting cannot reveal.
   - A published glucose accuracy (MAE 2.96 mg/dL) is matched only when
     augmentation precedes the split. In that setting a nearest-neighbour
     memoriser scores 0.1 mg/dL, and under a subject split the model does
     not beat the mean.
   - On 556 VitalDB surgical patients with arterial-line references, a
     random split lowers gradient-boosting MAE from 15.3 to 12.1 mmHg, almost
     entirely through demographic inputs that identify patients, and reverses
     which model looks best. The audit still confirms a genuine
     calibration-free waveform signal.

**Why TBME.** The work is an engineering method whose output changes what
the field can claim about cuffless sensors. It speaks directly to TBME
readers who develop or review such devices, and to the reporting standards
the journal applies.

**Reproducibility.** Every number in the paper is generated from released
code, frozen folds and out-of-fold predictions:
https://github.com/Namangarg484/IEEE-Gaurav-Verma. The datasets are public:
Kermani–Esmaeili (CC BY 4.0), BIG IDEAs (PhysioNet), STUBA ePPG (CC0),
PPG-BP (CC0) and VitalDB (CC BY 4.0).

The manuscript is original, has not been published, and is not under
consideration elsewhere. A preprint [has / has not] been posted on [arXiv].
The authors declare no competing interests. [Funding statement.]

**Suggested reviewers** (no conflicts; verify current affiliations before
submitting):
- [Name, affiliation, email]: PPG signal processing or cuffless blood pressure
- [Name, affiliation, email]: machine-learning evaluation or leakage
- [Name, affiliation, email]: non-invasive glucose sensing

Thank you for considering this submission.

Yours sincerely,

Gaurav Verma
[Department], [Institution]
[email]
