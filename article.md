# Robustness of Medical-Imaging Diagnosis under Adversarial Attacks: A Survey of Transformer-Based Models and Explainable AI

Hajar MAZOUZI Computer Science student  
University Hassan 1st  
Morocco mazouzi.ensa@uhp.ac.ma  

Abstract—Deep-learning systems are increasingly used to support diagnosis from radiographs, dermoscopic images, retinal scans and histopathology slides. Yet a perturbation that is invisible to the human eye can flip an automatic diagnosis, which makes the integrity of the whole diagnostic chain a cybersecurity issue. This survey reviews the literature on adversarial attacks against medical-image classifiers, with a specific focus on Transformer-based models (Vision Transformers, ViT) and on explainable AI (XAI), which is both a defensive tool and a new attack surface. We queried four open bibliographic sources (arXiv, Semantic Scholar, OpenAlex and Crossref), merged 3 021 records into 1 589 unique ones, and retained 45 publications from 2018–2026 following a PRISMA-style process. We propose a taxonomy of attacks (gradient-based, universal, patch/sparse, black-box transfer, poisoning/backdoor, and attacks on explanations) and of defenses (adversarial training, input purification, robust architectures, attack detection, XAI-based defenses), and compare the retained works by attack, model, XAI technique, defense and imaging domain. Our analysis shows that CNNs dominate the literature (18 of 45 papers) while only 17 address Transformers, and that only 2 papers combine Transformers and XAI, which reveals a clear gap. We also discuss datasets, evaluation metrics and open problems, and describe the design of a companion simulator that injects an imperceptible perturbation into a medical image, detects it, and visualizes the effect on both the diagnosis and the explanation.


Index Terms—Adversarial attacks, Medical imaging, Vision Transformer, Explainable AI, Robustness, Intrusion detection, FGSM, PGD, Cybersecurity, Healthcare


## I. INTRODUCTION


### A. Motivation and Context


Artificial intelligence is now embedded in clinical workflows: models flag pneumonia on chest radiographs, grade diabetic retinopathy on fundus photographs, classify skin lesions and score tumour slides. These systems promise faster and more consistent diagnoses, but they also create a new kind of asset to protect. A diagnosis produced by a model is trusted, archived in the patient record and used to decide treatment; whoever can silently alter it can cause harm without ever touching the patient’s data in a way that looks like a breach.


Szegedy et al. showed that small, carefully crafted perturbations can make a neural network misclassify an image [1], and Goodfellow et al. introduced a cheap way to compute them [2]. Finlayson et al. argued that medicine is a particularly attractive target for such attacks because of the financial incentives (insurance, reimbursement, drug and device approval) and the technical infrastructure through which images flow [3]. Mirsky et al. demonstrated that malicious tampering of 3D medical imagery, injecting or removing tumours in CT scans, can fool both radiologists and AI systems [4].


Several factors make the problem pressing for the healthcare ecosystem:

  1) High stakes. A misdiagnosis may delay a treatment or trigger an unnecessary one, so the cost of an undetected manipulation is measured in patient safety, not only in accuracy points.
  2) Connected infrastructure. Images travel across scanners, PACS archives, hospital networks, tele-radiology services and cloud platforms. Each hop is a place where a file can be altered, as studied for medical IoT devices [5].
  3) New model families. Vision Transformers [6] are replacing or complementing convolutional networks in medical imaging, but their robustness profile is different and still being established.
  4) Explainability as a safeguard, and as a target. XAI is widely promoted as the mechanism that lets clinicians check a model’s reasoning, while several works show that explanations can themselves be manipulated [7]– [10].
  5) Imperceptibility. Unlike a classical intrusion, the attack leaves no visible trace in the image, which calls for dedicated detection mechanisms.

### B. Problem Statement and Research Questions


In the cybersecurity vocabulary used throughout this work, the attacked asset is the integrity of the automatic diagnosis (the “I” of the CIA triad), and a defense is analogous to an intrusion detection/prevention system (IDS/IPS) placed in front of the model. The central question of this survey is:


To what extent do Transformer-based diagnostic models resist imperceptible adversarial perturbations, and can explainability help detect or mitigate such attacks? We break it down into five research questions:

- RQ1: Which attack families have been applied to medical-image classifiers and segmenters?
- RQ2: How do Transformer-based models compare with CNNs in terms of robustness?
- RQ3: Which defenses and detection approaches have been proposed, and how effective are they?
- RQ4: What role does XAI play, as a defense and as a target?
- RQ5: Which datasets and metrics are used, and which gaps remain open?

### C. Contributions


The main contributions of this article are:

- A PRISMA-style, reproducible collection of 45 publications (2018–2026) selected from 3 021 retrieved records, with the collection script made available.
- A taxonomy of attacks and a taxonomy of defenses for AI-based medical-image diagnosis.
- A comparative table of the 39 technical studies by attack, model, XAI technique, defense and imaging domain.
- An analysis of datasets and evaluation metrics, including metrics specific to explanation robustness and to detection (IDS view).
- The identification of a research gap at the intersection of Transformers, attacks and XAI.
- The design of a companion attack-and-detection simulator (architecture, use cases and algorithms) positioned against this gap.

### D. Paper Organization


Section II presents the background. Section III describes the review methodology and the PRISMA diagram. Sections IV and V present the taxonomies of attacks and of defenses. Section VI, the core of the survey, compares the retained works. Section VII discusses datasets and metrics. Section VIII discusses challenges, limitations and threats to validity, Section IX describes the simulator design, Section X gives future directions, and Section XI concludes.


## II. BACKGROUND


### A. AI-Assisted Medical Imaging and Its Attack Surface


1) Typical Pipeline: A diagnostic pipeline starts at the acquisition device (X-ray, CT, MRI, ultrasound, fundus camera, slide scanner), stores images in a PACS archive using the DICOM format, and sends them to an inference service that returns a label, a confidence score and, increasingly, an explanation map. The service may run on a hospital server or in the cloud. Fig. 1 shows the pipeline and the points where an adversary can act.


[Fig. 1. Medical-imaging AI pipeline and attack points. — voir le PDF]


2) Attack Surface: An adversary can act at several points: (i) on the network between the device, the PACS and the inference service (man-in-the-middle on image transfer); (ii) inside the hospital (a malicious insider or a compromised workstation); (iii) on a cloud inference API that is queried as a black box; (iv) on the training data (poisoning); and (v) on the explanation shown to the clinician. Our simulator focuses on cases (i) and (ii): a perturbation is added to an image before it reaches the model.


### B. Security Notions Mobilized

  1) CIA Triad:
- Confidentiality: not the main issue here, since no data is stolen, although model-extraction attacks used to craft black-box attacks may leak model information.
- Integrity: the property violated. The output no longer reflects the true content of the image, and nobody notices.
- Availability: preserved, since the system keeps answering; the attack is therefore stealthy. 2) IDS/IPS Analogy: An intrusion detection system observes events and raises alerts; an intrusion prevention system also blocks them. For adversarial inputs, the “event” is an image, the “signature” is a statistical or model-based anomaly, and the “prevention” is rejecting or purifying the input. Detection quality is therefore reported with the usual IDS metrics: detection rate, false-positive rate (FPR) and detection time.

### C. Adversarial Machine-Learning Notions


Let fθ be a classifier, x an image with true label y, and L the loss. An untargeted attack seeks a perturbation δ with ∥δ∥p ≤ϵ such that fθ(x + δ)̸ = y; a targeted attack forces a chosen label. The attacker model is white-box (full access to the model and its gradients), gray-box (partial knowledge) or black-box (only queries, or a surrogate model whose adversarial examples are transferred). Two reference gradient attacks are:


[équation — voir le PDF]


for the Fast Gradient Sign Method (FGSM) [2], and its iterative version


[équation — voir le PDF]


known as Projected Gradient Descent (PGD) [11], where Π projects back onto the ϵ-ball. Optimization-based attacks such as Carlini–Wagner (C&W) [12] and minimal-perturbation attacks such as DeepFool [13] are also widely used. Adversarial training, the main defense, solves the min–max problem


[équation — voir le PDF]


### D. Vision Transformers


A Vision Transformer splits an image into fixed-size patches, embeds them as tokens, adds position information and processes them with multi-head self-attention [6]. Unlike convolutions, self-attention is global from the first layer, which changes how perturbations propagate and offers an additional signal, the attention maps, that can be inspected or attacked. Several of the retained works exploit exactly this property [14], [15]. Transformers have also been adopted for the defense side, as in the Transformer-based attack detector of [16].


### E. Explainable AI


Post-hoc explanation methods produce a map of the image regions that support a prediction: Grad-CAM [17] uses gradients of the target class on convolutional feature maps, SHAP [18] and LIME [19] attribute the output to input features, and attention rollout aggregates the attention of Transformers. In an IDS perspective, explanations have two roles: a defensive one (an abnormal explanation can reveal an attack) and an offensive one (the attacker can target the explanation so that a wrong diagnosis still “looks right”).


### F. Related Surveys


Several surveys cover adversarial machine learning in healthcare. Qayyum et al. review the security and privacy of ML in healthcare applications [20]; Kaviani et al. [21] and a 2021 survey [22] focus on medical imaging informatics and robustness; a 2023 review analyses attack and defense for medical image analysis [23]; and Dong et al. propose a taxonomy by application scenario together with a benchmark of adversarially trained models [24]. Our survey differs by (i) its explicit focus on Transformers, (ii) its treatment of XAI both as defense and as target, and (iii) its IDS-oriented view of detection metrics.


## III. REVIEW METHODOLOGY


This review follows the PRISMA 2020 reporting guideline [25].


### A. Search Strategy


1) Sources: The brief suggests IEEE Xplore, ACM, Scopus and ScienceDirect. Because these platforms restrict automated access, we queried four open bibliographic APIs that index the same venues through DOI metadata: arXiv, Semantic Scholar, OpenAlex and Crossref. A Python script (provided with this work) runs all queries and exports titles, authors, year, venue, DOI, abstract and citation count.


[Fig. 2. PRISMA-style flow diagram of the selection process. — voir le PDF]


2) Search Strings: Ten queries combined three concept groups: adversarial/robustness, medical imaging and model/XAI. For example: “vision transformer adversarial medical images”, “adversarial attack explainable AI medical imaging”, “Grad-CAM adversarial perturbation medical diagnosis”, “FGSM PGD chest X-ray classifier”, “adversarial training medical imaging robustness”, “explainability robustness deep learning diagnosis attack”, “transformer medical image diagnosis security” and “adversarial examples radiology deep learning”.


### B. Inclusion and Exclusion Criteria

  1) Inclusion: 1) Published between 2018 and 2026.
  2) Addresses adversarial attacks, robustness or defenses.
  3) Concerns medical images (radiology, dermatology, ophthalmology, pathology, ultrasound, CT/MRI).
  4) Peer-reviewed article, conference paper or arXiv/medRxiv preprint with a usable abstract.
  2) Exclusion: 1) Generative adversarial networks used for image synthesis or augmentation without an attack or robustness goal.
  2) Non-imaging domains (text, signals, machine faults, smart grids).
  3) Errata, replies and retractions.
  4) Duplicates.

### C. Selection Process (PRISMA)


Fig. 2 shows the flow. Of the 3 021 records retrieved, 1 432 duplicates were removed (matching on DOI, then on normalised title). A keyword filter on title and abstract (medical term AND adversarial/robustness term, year 2018–2026) kept 294 records; a stricter relevance filter (imaging terms, removal of off-topic titles) kept 228 candidates. We then ranked the candidates by topical relevance (Transformer, XAI, attack, defense, survey) and citation count and retained 45 for indepth analysis.


### D. Data Extraction and Synthesis


For each retained paper we extracted: (i) metadata (authors, title, year, venue, DOI, citations); (ii) the attack family and


[TABLE I — DISTRIBUTION OF RETAINED PUBLICATIONS BY YEAR — voir le PDF]


[Fig. 3. Retained publications per year. — voir le PDF]


attacker knowledge; (iii) the model family (CNN, ViT, hybrid); (iv) the XAI technique if any; (v) the defense or detection approach if any; and (vi) the imaging domain. The extraction was first automated by keyword matching on title and abstract and then used to organize a qualitative thematic synthesis: codes (for example “FGSM on ViT”) were grouped into descriptive themes (attack families, defense families) and then into the taxonomies of Sections IV and V. Frequencies were counted to identify dominant trends and gaps.


### E. Period Covered and Retained Articles


The retained set covers 2018–2026 and contains 45 publications (6 surveys or reviews and 39 technical studies). Table I and Fig. 3 show the distribution by year: 25 of the 45 papers (56%) appeared in 2024–2026, which confirms that the topic is recent and growing.


## IV. TAXONOMY OF THREATS AND ATTACKS


### A. Classification of Attacks


Table II summarizes the attack families found in the literature on medical imaging, and Fig. 4 organizes them by the attacker’s knowledge and goal.


### B. Attack Vectors and Scenarios

- Man-in-the-middle on image transfer: the attacker intercepts the image between the scanner/PACS and the inference service and replaces it with a perturbed version. This is the scenario reproduced by our simulator.
- Insider or compromised workstation: a perturbation is added before upload, possibly to obtain a favourable (insurance) or unfavourable diagnosis [3].

[Fig. 4. Taxonomy of attacks (evasion attacks are further split into white-box and black-box/transfer settings). — voir le PDF]

- Cloud inference as a black box: the attacker has only query access and relies on transfer from a surrogate model [36].
- Tampering at the 3D-volume level: tumours injected into or removed from CT scans [4].
- Medical IoT devices: adversarial examples against COVID-19 deep-learning systems running on connected medical devices [5].
- Supply-chain and training-data attacks: poisoning of public or shared datasets [40]–[42].

### C. Impacts on Security and Safety


The consequences are a false negative (a missed lesion and delayed treatment), a false positive (unnecessary biopsy or treatment), insurance or reimbursement fraud, and a loss of clinician trust in the AI tool. Because the perturbation is imperceptible, the attack leaves no visible trace and can be repeated at scale. In the safety dimension, a manipulated segmentation can change a measured organ size or a radiotherapy contour [34], [35].


## V. TAXONOMY OF AI-BASED DEFENSE AND DETECTION APPROACHES


Fig. 5 presents the defense taxonomy. We distinguish defenses that act before the model (input-side), inside the model (model-side), around the model (detection) and through explanations (XAI-based).


### A. Input-Side Defenses


Preprocessing and purification try to remove the perturbation before inference. Examples in the corpus combine a Transformer feature extractor with diffusion-based generative modelling in an ensemble [44]; use diffusion purification, deep feature extraction and ensemble learning to cleanse samples and detect attacks on chest X-rays [45]; and apply segmentation and image enhancement so that a ViT is trained on critical features, reducing the impact of perturbations [46].


[TABLE II — TAXONOMY OF ATTACKS AGAINST AI-BASED MEDICAL-IMAGE DIAGNOSIS — voir le PDF]


[Fig. 5. Taxonomy of defenses against adversarial attacks in medical imaging. — voir le PDF]


### B. Model-Side Defenses


1) Adversarial Training: Adversarial training, i.e. training on adversarial examples, is the most common defense (9 of 45 papers). On a ViT for skin lesions, FGSM reduces accuracy from 90.1% to 27.38%, and adversarial training restores robustness [26]. Hybrid techniques combine adver- sarial training with handcrafted features (HOG, LBP, SIFT) to protect ViTs while limiting the loss of clean accuracy [29]. Adversarial training with feature fusion has been applied to diabetic retinopathy [47], and a dynamic perturbation-adaptive variant aims to preserve generalization while improving robustness [48]. Adversarial training also affects explanations: it yields sharper and more coherent saliency maps on skincancer CNNs [49], and robust models give explanations better aligned with clinically meaningful regions in fracture detection [50]. Similarly, robust models were found to give more stable predictions and improved interpretability for ultrasound-based COVID-19 diagnosis [51].


2) Robust Architectures and Ensembles: Robust ViT designs and architecture comparisons (CNN, ViT, hybrid) appear mostly in 2024–2026 [27], [52]–[55]. A proposed robust ViT achieves better robustness than a standard ViT, at the price of a lower accuracy on clean images [54], which illustrates the robustness–accuracy trade-off. Combining explainable feature-based defense, adversarial training with transfer learning and classifier augmentation is proposed in [56].


### C. Attack Detection


Detection treats the input like network traffic: a detector flags suspicious images before they reach the classifier, which is the IDS/IPS analogy of the project brief. RAD-IoMT proposes a Transformer-based attack detector for medical images in IoMT environments [16]; RADAR-MIX uses quantitative metrics and visual analysis to uncover attacks instead of a binary decision [31]; and a 2026 framework fuses feature analysis, anomaly detection and interpretability to detect adversarial inputs and link them to explanations [57].


### D. XAI-Based Defenses


Explanation maps can be used to refine a model (Grad- CAM-guided feature masking against FGSM and PGD [30]) or to expose an attack (inconsistency between the prediction and the explanation [8], [31]). The reverse also holds: explanations must themselves be evaluated for robustness [9].


## VI. COMPARATIVE STUDY


### A. Analysis by Attack Type and by Model


1) Gradient-Based Attacks on Transformers: Kanca et al. report that a ViT for skin-lesion classification drops from 90.1% to 27.38% accuracy under FGSM, and that adversarial training mitigates the loss [26]. SPARTA is a white-box, attention-guided sparse patch attack evaluated on ISIC2018, ChestX-ray14, OCT2017 and PathMNIST with ViT-B/16, DeiT-S and ResNet50, reaching attack success rates of up to 89.5% while perturbing only a small fraction of the image [14]. A study of adversarial watermarking investigates ViTs for dermatological images and the transfer of the perturbations to CNNs [15].


2) CNN versus Transformer Comparisons: A 2026 study evaluates a CNN, VGG19 and a ViT for chest X-ray classification under FGSM and iterative FGSM with perturbation levels from ϵ = 0 to 0.30, and compares five defense strategies under identical conditions [27]. Other comparisons cover CNN, Transformer and hybrid architectures [55]. A multi-metric evaluation argues that attack success rate alone is insufficient and that perturbation strength, perceptual quality and cross-architecture transferability must also be reported, including for ViTs [38].


3) Universal and Black-Box Attacks: Universal perturbations fool classifiers for skin cancer, referable diabetic retinopathy and pneumonia across seven architectures [33]. In black-box settings, the factors that influence transfer (weight initialization, pre-training and architecture) have been analysed in three medical domains [36]. In computational pathology, both white- and black-box attacks confuse CNNs, and adversarial training improves resistance [37].


4) Segmentation and Retrieval: Beyond classification, adversarial examples can force anatomically implausible segmentations in chest X-rays [34] or deceive U-Net CT segmentation models while keeping a low perceptibility [35]. In medical image retrieval, a Vision Transformer hashing model with a residual fuzzy generator has been proposed for targeted attacks [39].


5) Attacks and Defenses Involving Explanations: Rasaee and Rivaz show that practically undetectable perturbations can dramatically alter Grad-CAM importance maps for breastultrasound classification [8]. A black-box attack misleads


[Fig. 6. Imaging domains mentioned in the retained papers (a paper can cover several domains). — voir le PDF]


image interpreters such as Grad-CAM using the Sep-CMA- ES optimizer [10]. Repetto et al. evaluate the stability of XAI methods under natural noise and adversarial manipulations that change explanations without changing predictions [9]. Conversely, Grad-CAM-guided refinement can improve robustness [30], and explanation-based analysis can reveal attacks [31].


6) Transformer-Based Detection and Robust Ensembles: Transformer-based detection is explored for IoMT medical images [16], and a Transformer–diffusion ensemble targets reliability under corruption and adversarial noise [44].


### B. Comparison by Imaging Domain


Chest X-ray is the most studied domain (12 papers), followed by dermatology (11), ophthalmology (6), pathology (3) and ultrasound (2) (Fig. 6). Dermatology and chest X-ray benefit from large public datasets (HAM10000/ISIC, ChestX-ray14, CheXpert), which explains their dominance. Pathology and 3D modalities (CT, MRI) are less covered in the Transformer literature, although whole-slide and volumetric data are an attractive target.


### C. Summary Comparison Table


Table III lists the 39 technical studies. The columns were extracted automatically from the title and abstract by keyword matching; a dash means that the item is not mentioned in the abstract, not that it is absent from the full paper, so entries should be confirmed on the full text before being relied on.


### D. Trends and Observations


Beyond the six surveys [20]–[24], [57], we observe the following (Fig. 7):

- CNNs still dominate: 18 papers mention CNNs and 17 mention Transformers; only 9 combine a Transformer with an attack.
- XAI is present but rarely combined with Transformers: 19 papers mention explainability, 9 combine it with an attack, and only 2 combine Transformers with XAI.
- Defenses: adversarial training appears in 9 papers and attack detection in 5; detection is therefore the less explored direction.
- Domains: chest X-ray (12 papers) and dermatology (11) are the most frequent.

[TABLE III — COMPARISON OF TECHNICAL STUDIES BY ATTACK, MODEL, XAI, DEFENSE AND DOMAIN (AT = ADVERSARIAL TRAINING; CXR = CHEST X-RAY; — voir le PDF]


[Fig. 7. Number of retained papers per theme. — voir le PDF]

- Recency: 25 of 45 papers are from 2024–2026, and many are preprints or short conference papers, so results should be reproduced before being generalised.

## VII. DATASETS AND EVALUATION METRICS


### A. Public Datasets


Table IV lists the main public datasets used in the field. The ISIC2018, ChestX-ray14, OCT2017 and PathMNIST datasets


[TABLE IV — PUBLIC DATASETS COMMONLY USED IN ADVERSARIAL-ROBUSTNESS — voir le PDF]


are, for instance, jointly used in [14].


### B. Evaluation Protocols


A fair protocol specifies (i) the threat model (white/gray/black-box), (ii) the norm and budget ϵ (for example L∞with ϵ in pixel-intensity units), (iii) the number


[TABLE V — EVALUATION METRICS — voir le PDF]


of attack iterations, (iv) whether the attack adapts to the defense, and (v) standardized attack suites such as AutoAttack [64]. Black-box results must state how the surrogate model was obtained [36]. A sweep of budgets rather than a single value, as in [27], gives a robustness curve instead of a single number.


### C. Metrics


Table V groups the metrics. For detection, the IDS vocabulary (detection rate, FPR, detection time) is the one requested by the project brief.


## VIII. DISCUSSION


### A. Challenges and Open Problems

- Realism of the threat model: many studies are white-box with arbitrary budgets, while a hospital attacker is more likely to work in a black-box or gray-box setting.
- Robustness–accuracy trade-off: adversarial training costs time and may reduce clean accuracy [29], [54].
- Fragility of explanations: explanations can be manipulated [7]–[10], so XAI cannot be trusted blindly as a safeguard.
- Comparability: heterogeneous datasets, budgets and metrics make direct comparison difficult [38].
- Computational cost: purification with diffusion models and large ensembles may be too heavy for real-time clinical use [44], [45].
- Generalization: most defenses are validated on one dataset or modality.

### B. Limitations of Current Approaches


Most defenses are tested against the attacks they were designed for, and adaptive attackers who know the defense are rarely considered. Detection is explored in only 5 of the 45 papers, and explanation-based detection is even rarer. Robustness benchmarks mostly address classification, whereas segmentation and retrieval are also vulnerable [34], [35], [39].


### C. Positioning of the Contribution and of the Realization


The comparison shows few studies that jointly study Transformers, attacks and explanations. Our companion simulator, described in Section IX, targets this gap by showing the diagnosis, the explanation and the detection decision side by side, before and after an attack.


[Fig. 8. Architecture of the proposed simulator. — voir le PDF]


### D. Threats to Validity


1) Selection Bias: Open APIs rather than IEEE, ACM, Scopus or ScienceDirect were used, so coverage may differ from a search on those platforms. The ranking step that led to 45 papers favours highly cited and recent works. 2) Extraction Accuracy: The tagging in Table III is derived from titles and abstracts; some abstracts were missing in the retrieved metadata, and paywalled full texts were not analysed. Entries must be verified on the full text.


3) Single-Reviewer Process: Selection and extraction were performed by a single author, without independent verification or inter-rater agreement, which increases the risk of subjective classification.


4) Temporal Limits: The field evolves quickly, and several recent items are preprints that may change after peer review.


## IX. DESIGN OF THE COMPANION SIMULATOR


This section describes the design of the simulator associated with the survey, following the realization guide of the project. Its experimental evaluation is left for future work.


### A. Problem and Objectives


The problem is to let a user observe how an imperceptible perturbation changes an automatic diagnosis and its explanation, and whether a detector can catch it. The objectives are: (O1) load a medical image; (O2) classify it with a Vision Transformer; (O3) inject an FGSM or PGD perturbation with adjustable ϵ; (O4) display the original image, the attacked image, the amplified perturbation, and the diagnosis and confidence before and after; (O5) display the explanation map (Grad-CAM or attention rollout) before and after; (O6) run a detector and report detection rate, FPR and detection time.


### B. Architecture


Fig. 8 shows a three-tier architecture: a web interface (PHP), a Python inference service exposing a REST API, and a storage layer for images, results and logs. The Python service hosts the ViT classifier, the attack module, the XAI module and the detector, because PHP cannot run these models.


### C. Use Cases


The actors are the user (analyst or student) and the system. The main use cases are: upload an image, run the classification, choose an attack and its budget, inject the attack, view the


[Algorithm 1 Attack injection (PGD / FGSM) — voir le PDF]


iterations T 1: x0 ←x 2: for t = 0 to T −1 do


[Algorithm 2 Detector — voir le PDF]


comparison, run the detector, view the XAI map, and export a report. An attacker is modelled by the simulation module, not as a human actor.


### D. Algorithms


Algorithm 1 describes the attack injection (PGD; FGSM is the special case with one step and α = ϵ), and Algorithm 2 the detector logic that combines a feature-based anomaly score with an explanation consistency check, in line with the detection approaches of Section V.


### E. Test View


The planned test scenarios use public datasets (Table IV), clean and attacked sets for several ϵ values, and report accuracy under attack, attack success rate, detection rate, FPR, F1 and detection time (Table V).


## X. FUTURE DIRECTIONS


### A. Short Term (1–2 years)

  1) Systematic benchmarks of ViT, CNN and hybrid models under the same threat model and metrics.
  2) Standardized reporting of budgets, attack iterations and datasets.
  3) Open-source simulators for education and testing.

### B. Medium Term (3–5 years)

  1) Joint evaluation of attacks on the prediction and on the explanation, including adaptive attacks against XAI-based detectors.
  2) Lightweight, real-time detectors suited to hospital networks (IDS/IPS integration at the PACS gateway).
  3) Defenses for federated and multi-centre settings, and robustness to poisoning of training data.

### C. Long Term (5+ years)

  1) Certified robustness guarantees for diagnostic models.
  2) Human studies measuring whether clinicians actually detect manipulated images or explanations.
  3) Regulatory frameworks that require adversarial testing before clinical deployment.

## XI. CONCLUSION


### A. Summary of Contributions


This survey reviewed 45 publications (2018–2026) on adversarial attacks and defenses for medical-image diagnosis, with a focus on Transformers and XAI. We proposed taxonomies of attacks and of defenses, compared the retained works, and described the design of a companion simulator.


### B. Key Takeaways

- The threat is real: imperceptible perturbations can reduce the accuracy of a ViT from 90.1% to 27.38% on skin lesions [26].
- CNNs still dominate the literature, and Transformers are less studied (17 papers).
- Detection is under-studied compared with adversarial training (5 versus 9 papers).
- Explanations are both a defense and a target, and very few works combine Transformers, attacks and XAI (2 papers).
- Evaluation is heterogeneous; reporting budgets, transferability and detection metrics would improve comparability.

### C. Final Remarks


The integrity of AI-assisted diagnosis should be treated as a cybersecurity requirement, with defenses and monitoring comparable to those used for networks. Future work will implement and evaluate the simulator and complete the comparison with full-text verification of each study.


## ACKNOWLEDGMENTS


The author thanks the researchers whose work was analysed in this survey.


## REFERENCES


[1] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow,

and R. Fergus, “Intriguing properties of neural networks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2014. [2] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Proc. Int. Conf. Learn. Representations (ICLR), 2015. [3] S. G. Finlayson, J. D. Bowers, J. Ito, J. L. Zittrain, A. L. Beam, and I. S.


Kohane, “Adversarial attacks on medical machine learning,” Science, vol. 363, no. 6433, pp. 1287–1289, 2019. [4] Y. Mirsky, T. Mahler, I. Shelef, and Y. Elovici, “CT-GAN: Malicious tampering of 3D medical imagery using deep learning,” in Proc. 28th USENIX Security Symp., 2019, pp. 461–478. [5] M. A. Rahman, M. S. Hossain, N. A. Alrajeh, and F. J. Alsolami,


“Adversarial examplessecurity threats to covid-19 deep learning systems in medical iot devices,” IEEE Internet of Things Journal, 2020. [6] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Representations (ICLR), 2021. [7] A. Ghorbani, A. Abid, and J. Zou, “Interpretation of neural networks is fragile,” in Proc. AAAI Conf. Artificial Intelligence, vol. 33, 2019, pp. 3681–3688. [8] H. Rasaee and H. J. Rivaz, “Explainable ai and susceptibility to adversarial attacks: a case study in classification of breast ultrasound images,” in IEEE International Ultrasonics Symposium, 2021. [9] S. Repetto, I. Maljkovic, M. Lotto, A. E. Cinà, S. Vascon, and F. Roli,


“Evaluating the robustness of explainable ai in medical image recognition under natural and adversarial data corruption,” Machine Learning, 2025. [10] Y. Hirose and S. Ono, “Black-box adversarial attack against visual interpreters for deep neural networks,” Preprint, 2023. [11] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2018. [12] N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” in Proc. IEEE Symp. Security and Privacy, 2017, pp. 39–57. [13] S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “DeepFool: A simple and accurate method to fool deep neural networks,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2574–2582. [14] T. D. Thanh, T. T. T. Pham, and H.-G. Doan, “Sparta: Sparse patch attack for revealing transformer attention vulnerabilities,” Engineering Technology & Applied Science Research, 2026. [15] R. Sadik, T. Rahman, A. Bhattacharjee, B. C. Halder, I. Hossain,


M. Banik et al., “Exploring adversarial watermarking in transformer-based models: Transferability and robustness against defense mechanism for medical images,” arXiv, 2025. [16] S. Rahman, S. Pal, A. Fallah, R. Doss, and C. K. Karmakar, “Rad- iomt: Robust adversarial defence mechanisms for iomt medical image analysis,” Ad Hoc Networks, 2025. [17] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and


D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626. [18] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017. [19] M. T. Ribeiro, S. Singh, and C. Guestrin, ““why should I trust you?”:


Explaining the predictions of any classifier,” in Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 1135–1144. [20] A. Qayyum, J. Qadir, M. Bilal, and A. AlFuqaha, “Secure and robust machine learning for healthcare: A survey,” IEEE Reviews in Biomedical Engineering, 2020. [21] S. Kaviani, K. J. Han, and I. Sohn, “Adversarial attacks and defenses on ai in medical imaging informatics: A survey,” Expert Systems with Applications, 2022. [22] K. D. Apostolidis and G. A. Papakostas, “A survey on adversarial deep learning robustness in medical image analysis,” Electronics, 2021. [23] G. W. Muoka, Y. Ding, C. C. Ukwuoma, A. Mutale, C. J. Ejiyi, A. K.


Mzee et al., “A comprehensive review and analysis of deep learningbased medical image adversarial attack and defense,” Mathematics, 2023.


[24] J. Dong, J. Chen, X. Xie, J. Lai, and H. Chen, “Survey on adversarial

attack and defense for medical image analysis: Methods and challenges,” ACM Computing Surveys, 2024. [25] M. J. Page et al., “The PRISMA 2020 statement: An updated guideline for reporting systematic reviews,” BMJ, vol. 372, p. n71, 2021. [26] E. Kanca, S. Ayas, E. B. Kablan, and M. Ekinci, “Implementation of fast gradient sign adversarial attack on vision transformer model and development of defense mechanism in classification of dermoscopy images,” in Signal Processing and Communications Applications Conference (SIU), 2023. [27] Y. Heryadi, I. Sonata, and B. K. Triwijoyo, “Comparative analysis of adversarial robustness across cnn, vgg19, and vision transformer architectures using multiple defense strategies,” Matrik Jurnal Manajemen Teknik Informatika dan Rekayasa Komputer, 2026. [28] O. Daanouni, B. Cherradi, and A. Tmiri, “Nsl-mha-cnn: A novel cnn architecture for robust diabetic retinopathy prediction against adversarial attacks,” IEEE Access, 2022. [29] A. S. Neha, V. Chaturvedi, and M. Shafique, “Hy-deft: A hybrid defense technique for vision transformers against adversarial attacks in medical imaging,” in International Joint Conference on Neural Networks (IJCNN), 2025. [30] L. Wang, I. I. Uddin, X. Qin, Y. Zhou, and K. Santosh, “Explainability- driven defense: Grad-cam-guided model refinement against adversarial threats,” in Proceedings of the AAAI Symposium Series, 2025. [31] E. J. de Aguiar, C. Traina, and A. J. M. Traina, “Radar-mix: How to uncover adversarial attacks in medical image analysis through explainability,” in Proceedings - IEEE Symposium on Computer-Based Medical Systems, 2024. [32] S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, “Univer- sal adversarial perturbations,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1765–1773. [33] H. Hirano, A. Minagi, and K. Takemoto, “Universal adversarial attacks on deep neural networks for medical image classification,” BMC Medical Imaging, 2021. [34] G. Bortsova, F. Dubost, L. Hogeweg, I. Katramados, and M. de Bruijne,


“Adversarial heart attack: Neural networks fooled to segment heart symbols in chest x-ray images,” arXiv, 2021. [35] W. Lee, M. Ju, Y. Sim, Y. K. Jung, T. H. Kim, and Y. Kim, “Adversarial attacks on medical segmentation model via transformation of feature statistics,” Applied Sciences, 2024. [36] G. Bortsova, C. González-Gonzalo, S. C. Wetstein, F. Dubost, I. Katra- mados, L. Hogeweg et al., “Adversarial attack vulnerability of medical image analysis systems: Unexplored factors,” Medical Image Analysis, 2021. [37] N. G. Laleh, D. Truhn, G. P. Veldhuizen, T. Han, M. van Treeck,


R. D. Buelow et al., “Adversarial attacks and adversarial robustness in computational pathology,” Nature Communications, 2022. [38] E. Curl, K. Ampomah, M. Erfan, and S. V. Dibbo, “Beyond attack success rate: A multi-metric evaluation of adversarial transferability in medical imaging models,” Preprints.org, 2026. [39] W. Ding, C. Liu, J. Huang, C. Cheng, and H. Ju, “Vith-rfg: Vision transformer hashing with residual fuzzy generation for targeted attack in medical image retrieval,” IEEE Transactions on Fuzzy Systems, 2023. [40] M. S. Nwadike, T. Miyawaki, E. Sarkar, M. Maniatakos, and F. E.


Shamout, “Explainability matters: Backdoor attacks on medical imaging,” arXiv, 2020. [41] S. B. Lee, “Gradual poisoning of a chest x-ray convolutional neural network with an adversarial attack and ai explainability methods,” Scientific Reports, 2025. [42] L. M. D. P, M. Gunasekaran, P. S, R. Samikannu, A. Yahya, I. A.


Badruddin et al., “Invisible poison attacks and holistic defence strategy for pyramid vision transformers in medical imaging,” Scientific Reports, 2026. [43] W. Li and W. M. N. W. Zainon, “Dual adversarial attacks on explainable deep learning in medical image classification,” Computer Methods and Programs in Biomedicine, 2026. [44] X. Shen, H. Huang, B. Nichyporuk, and T. Arbel, “Improving robustness and reliability in medical image classification with latent-guided diffusion and nested-ensembles,” IEEE Transactions on Medical Imaging, 2025. [45] A. K. Ali, “Robust and interpretable chest x-ray classification via diffusion purification and concept-based adversarial detection,” Basrah Researches Sciences, 2025.


[46] A. S. Neha, V. Chaturvedi, and M. Shafique, “S-e pipeline: A vision

transformer (vit) based resilient classification pipeline for medical imaging against adversarial attacks,” in International Joint Conference on Neural Networks (IJCNN), 2024. [47] S. Lal, S. U. Rehman, J. H. Shah, T. Meraj, H. T. Rauf, R. Damasevicius et al., “Adversarial attack and defence through adversarial training and feature fusion for diabetic retinopathy recognition,” Sensors, 2021. [48] S. Li, X. Ma, S. Jiang, and L. Meng, “Dynamic perturbation-adaptive adversarial training on medical image classification,” arXiv, 2024. [49] A. Margeloiu, N. Simidjievski, M. Jamnik, and A. Weller, “Improving interpretability in medical imaging diagnosis using adversarial training,” arXiv, 2020. [50] M. H. Najafi, M. Morsali, M. Pashanejad, S. S. Roudi, M. Norouzi, and S. B. Shouraki, “Secure diagnostics: Adversarial robustness meets clinical interpretability,” arXiv, 2025. [51] J. Roberts and T. Tsiligkaridis, “Ultrasound diagnosis of covid-19:


Robustness and explainability,” arXiv, 2020. [52] E. Kanca, S. Ayas, E. B. Kablan, and M. Ekinci, “Enhancing the adversarial robustness in medical image classification: exploring adversarial machine learning with vision transformers-based models,” Neural Computing and Applications, 2024. [53] ——, “Evaluating and enhancing the robustness of vision transformers against adversarial attacks in medical imaging,” Medical & Biological Engineering & Computing, 2024. [54] E. Kanca, T. Gulsoy, S. Ayas, and E. B. Kablan, “Robust vision transformer model against adversarial attacks in medical image classification,” in International Conference on Telecommunications and Signal Processing (TSP), 2024. [55] E. Altarawneh and E. Alshdaifat, “Adversarial robustness of cnn, trans- former, and hybrid architectures in medical imaging ai,” in Lecture notes in electrical engineering, 2026. [56] A. Jain, K. Sangeeta, B. H. Swathi, M. Sadim, S. P. Dwivedi, and


A. Albawi, “The impact of adversarial attacks on medical imaging ai systems,” in IEEE International Conference on Communication Systems and Network Technologies (CSNT), 2024. [57] H. Khamar, D. S. Degadwala, and D. Soni, “A systematic review on ad- versarial attack detection in medical image classification,” International Journal of Scientific Research in Science Engineering and Technology, 2026. [58] M. Paschali, S. Conjeti, F. Navarro, and N. Navab, “Generalizability vs.


robustness: Investigating medical imaging networks using adversarial examples,” in Lecture notes in computer science, 2018. [59] X. Ma, Y. Niu, L. Gu, Y. Wang, Y. Zhao, J. Bailey et al., “Understanding adversarial attacks on deep learning based medical image analysis systems,” Pattern Recognition, 2021. [60] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers,


“ChestX-ray8: Hospital-scale chest X-ray database and benchmarks,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2097–2106. [61] J. Irvin et al., “CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,” in Proc. AAAI Conf. Artificial Intelligence, vol. 33, 2019, pp. 590–597. [62] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, p. 180161, 2018. [63] D. S. Kermany et al., “Identifying medical diagnoses and treatable diseases by image-based deep learning,” Cell, vol. 172, no. 5, pp. 1122– 1131, 2018. [64] F. Croce and M. Hein, “Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,” in Proc. Int. Conf. Machine Learning (ICML), 2020, pp. 2206–2216.


## AUTHOR BIOGRAPHY


Hajar Mazouzi is a Computer Science student at the University Hassan 1st, Morocco. Her academic interests include cybersecurity, artificial intelligence and the security of AI systems in healthcare. Contact: mazouzi.ensa@uhp.ac.ma.
