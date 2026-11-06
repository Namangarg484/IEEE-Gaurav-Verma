# Pre-submission enquiry to the Editor-in-Chief, IEEE Transactions on Biomedical Engineering

Send this before submitting. TBME's author information page lists the
current Editor-in-Chief and the contact address; check it on the day you
send, because editors rotate. Attach the abstract (below) and, optionally,
the PDF.

---

**Subject:** Pre-submission enquiry: a validated leakage audit for PPG-based physiological estimation

Dear Professor [Editor-in-Chief's surname],

I would be grateful for your view on whether the manuscript summarised below
falls within the scope of IEEE Transactions on Biomedical Engineering before
I submit it.

The manuscript presents an evaluation method for physiological regression
from wearable photoplethysmography (PPG), validated as an engineering method
in its own right. The method detects labels that were corrupted while a
public table was built, flags augmented copies that cross a train/test split,
measures how much each leakage mechanism inflates a metric, and decides
whether a model beats a null model using subject-cluster bootstrap intervals.
We validate each component on synthetic data of known ground truth and then
apply the method to blood glucose and blood pressure:

- It finds a previously unreported label-replication error in a widely used
  public glucose table.
- It matches a published glucose accuracy only when data augmentation
  precedes the train/test split. In that setting a nearest-neighbour
  memoriser scores better than the published model, and the model does not
  beat the mean once subjects are separated.
- On 556 surgical patients from VitalDB with arterial-line references, it
  measures how much a random split inflates cuffless blood-pressure accuracy
  for flexible models, while confirming a genuine calibration-free signal.

All code, frozen folds and predictions are public:
https://github.com/Namangarg484/IEEE-Gaurav-Verma

The work fits TBME's emphasis on methods with demonstrated biomedical
significance, because evaluation errors of this kind directly affect claims
of device-level accuracy for cuffless sensing. I would value your opinion on
whether TBME is an appropriate venue, or whether you would recommend a
sister journal.

Thank you for your time.

Yours sincerely,
Gaurav Verma
[Department], [Institution]
[email]

---

**Abstract** (paste the current structured abstract from `tbme_paper.pdf`.)
