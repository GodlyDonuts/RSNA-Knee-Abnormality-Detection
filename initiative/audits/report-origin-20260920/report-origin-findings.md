# Report origin and cartilage grading

Source provenance could resolve an ambiguous grading convention if a report can be linked to a center and that center's reporting standard is documented. The released evidence checked here does not establish that link.

RSNA's current challenge page describes 19 sites and approximately 12 report languages. Its August 5 press release described 16 sites, while the current acknowledgment PDF lists 19 data contributors plus three additional sites. These are published descriptions, not a study-to-center mapping, and none establishes how many centers are represented in the 4,407 released training studies.

- [Current RSNA challenge description](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [RSNA August 5 press release](https://www.rsna.org/media/press/2026/2669)
- [RSNA contributor list](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/2026-rsna-ai-challenge_knee-challenge-acknowledgment-page.pdf?hash=15BC97C9FCC91E33D0CC2D2C5D853F9F&rev=0e928b87a06b4fc4a769d9dcdb298ff0)

## Released metadata checked

The authenticated training CSV contains a study identifier, report, and twelve target columns; there is no center or country column. The organizers say DICOM metadata was restricted to an allowlist of 86 tags. [Kaggle data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

An existing audit examined one verified DICOM per released study. Its scanner metadata contains 46 model strings and 51 manufacturer/model pairs. This audit fetched 53 independently hash-verified source DICOMs: one study covering each pair, plus all five batch-000 cartilage-grade candidates (some overlap the pair representatives).

Every sampled header lacked InstitutionName, InstitutionAddress, InstitutionalDepartmentName, StationName, DeviceSerialNumber, ClinicalTrialSiteID, ClinicalTrialSiteName, and IssuerOfPatientID. Manufacturer, model, field strength, software version, and some sequence descriptions survive. These findings cover the 53 selected headers, not every DICOM.

## Consequence for the five ambiguous reports

Three Spanish-language candidates have compatible report wording; two share a Skyra model/software combination and the third uses a Lumina. The English-language candidate uses a Vida and the Croatian-language candidate an Ingenia. These are acquisition and report-style clues. Shared language, template, or scanner does not establish a shared center or grading system. The acknowledged contributor countries also do not provide a reliable per-report linkage.

No center, country, or cartilage grade interpretation was assigned from these clues. The separate semantic adjudication remains grounded in each original report; missing grading context stays uncertain until adequately supported.

## Use in training preparation

Retain observed scanner information and clearly labeled report-style/language cohorts for later coverage and error analysis. Check whether uncertainty, reviewed label distributions, image quality, or performance vary across those cohorts. Do not describe these proxy cohorts as validated centers or claim a true center-held-out split. Report text is unavailable at test time, so reporting-cohort analysis is an audit tool, not an inference input.

The full image audit and independent report reviews remain in progress. This investigation changes neither the frozen clinical rubric nor the original annotations.
