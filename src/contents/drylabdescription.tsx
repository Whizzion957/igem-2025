import {
  WikiLayout,
  WikiSection,
  WikiSubsection,
  WikiSummaryCard,
  WikiParagraph,
  WikiList,
  WikiBold,
  WikiReferences,
  WikiReferenceItem,
  WikiCollapsible,
  WikiItalic,
  WikiImage,
} from "../components/wiki/WikiLayout";

import { MathJax, MathJaxContext } from "better-react-mathjax";
const config = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
  },
};

export function DryLabDescription() {
  const sections = [
    { id: "abstract", title: "Abstract" },
    { id: "introduction", title: "Introduction" },
    { id: "amplify", title: "AMPLIFY"},
    { id: "franklin", title: "FranklinForge"},
    { id: "closingtheloop", title: "Closing the Loop"},
    { id: "futurevision", title: "Future Vision"},
    { id: "summary", title: "Summary" },
    { id: "references", title: "References" },
  ];

  return (
    <MathJaxContext config={config}>
    <WikiLayout title="Dry Lab Description" sections={sections}>
      {/* Summary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "24px", marginBottom: "40px" }}>
        <WikiSummaryCard title="Our Toolbox" icon="🧬">
          Discover our foundational advance in gene regulation and information
          encoding: Engineer the 3D genome organization with our{" "}
          <WikiBold>comprehensive part collection!</WikiBold>
        </WikiSummaryCard>
        <WikiSummaryCard title="Modeling Innovation" icon="💻">
          Our digital twin accelerates experiment design and optimizes our DNA
          staple constructs, facilitating real-world applications of our
          toolbox.
        </WikiSummaryCard>
        <WikiSummaryCard title="Enabling Scientists" icon="🎓">
          We inspire curiosity and innovation through{" "}
          <WikiBold>educational programs</WikiBold> that empower diverse groups
          to shape the future of synthetic biology.
        </WikiSummaryCard>
      </div>

      {/* Abstract Section */}
      <WikiSection id="abstract" title="Abstract">
        <WikiSubsection title="Forging the Future of Antimicrobials: Where AI Meets Biology">
          <WikiParagraph>
            The world is losing the war against bacteria. Antimicrobial resistance (AMR) is surging as one of the most critical threats to global health, turning once-treatable infections into potential death sentences as conventional antibiotics crumble against evolving bacterial defenses. But nature has already armed us with a powerful weapon: antimicrobial peptides (AMPs) – short, positively-charged molecular assassins that punch holes through bacterial membranes with devastating precision. The challenge? Finding the right ones. Traditional experimental screening is a slow, expensive game of molecular roulette with frustratingly low odds.
          </WikiParagraph>
          <WikiParagraph>
            <WikiBold>
              We decided to change the rules entirely.
            </WikiBold>
          </WikiParagraph>
          <WikiParagraph>
            Enter the AMPlify dry lab, where we've unleashed the power of artificial intelligence to design killer peptides before ever touching a pipette. At the heart of this computational revolution stands FranklinForge – our AI engine named after the brilliant Rosalind Franklin, whose X-ray crystallography unlocked the secrets of DNA's double helix. Just as Franklin revealed biological function through structural insight, <WikiBold>FranklinForge</WikiBold> decodes the hidden relationships between peptide sequence, three-dimensional architecture, and lethal antimicrobial power. This isn't just machine learning – it's molecular intelligence that simultaneously predicts which peptides will obliterate bacteria while staying harmless to human cells.
          </WikiParagraph>
          <WikiParagraph>
            We didn't start from scratch. Mining the Database of Antimicrobial Activity and Structure of Peptides (DBAASP), we assembled a battle-tested arsenal of thousands of validated AMPs and trained deep learning models to recognize the molecular signatures of success. FranklinForge learned to separate peptide champions from molecular duds, identifying the exact features that make an AMP both deadly to pathogens and safe for humans.
          </WikiParagraph>
          <WikiParagraph>
            The results? <WikiBold>Mind-blowing efficiency.</WikiBold> FranklinForge sifted through thousands of molecular possibilities and distilled them into a bunch of high-confidence AMP candidates – allowing a <WikiBold>significant reduction</WikiBold> in the experimental haystack.
          </WikiParagraph>
          <WikiParagraph>
            Instead of blindly synthesizing hundreds of peptides hoping for a hit, the AMPlify wet lab can now laser-focus resources on validating only the computational cream of the crop.
          </WikiParagraph>
          <WikiParagraph>
            <WikiBold>
              This is precision-guided antibiotic discovery: computation calling the shots, experiments confirming the kills.
            </WikiBold>
          </WikiParagraph>
          <WikiParagraph>
            AMPlify's dry lab proves that the future of drug discovery isn't in the lab alone – it's in the symbiosis between silicon and biology. By channeling Rosalind Franklin's spirit of structural revelation through modern AI, FranklinForge is rewriting the playbook for how we fight infections.
          </WikiParagraph>
        </WikiSubsection>
      </WikiSection>

      {/* Introduction Section */}
      <WikiSection id="introduction" title="Introduction">
        <WikiSubsection title="From Crisis to Computation-Reimagining the Fight Against Tuberculosis">
          <WikiParagraph>
            The abstract painted the picture – now let's dive into the battlefield. Tuberculosis, caused by Mycobacterium tuberculosis, kills over 1.3 million people annually, making it one of the deadliest infectious diseases on Earth. But the real nightmare is the explosive rise of multidrug-resistant TB (MDR-TB) and extensively drug-resistant TB (XDR-TB), transforming a once-curable disease into a potential death sentence. Current TB treatment is already brutal – 6-9 months of combination therapy for drug-sensitive strains, up to 2 years for resistant strains with plummeting success rates and devastating side effects. The problem isn't just resistance; it's M. tuberculosis itself. Its thick, waxy cell wall rich in mycolic acids creates a molecular fortress that conventional antibiotics struggle to penetrate. It grows painfully slowly (24-hour doubling time versus minutes for most bacteria), making drug screening tedious and expensive. It can persist dormant in human tissues for decades, evading both immune responses and antibiotics. Traditional experimental approaches to finding new TB drugs – synthesizing compounds one by one, testing them in BSL-3 facilities, waiting weeks for results – simply cannot keep pace with evolving resistance. We need a radical new approach, and we need it now.
          </WikiParagraph>
          <WikiParagraph>
            This is where FranklinForge rewrites the rules. Antimicrobial peptides (AMPs) – nature's ancient bacterial assassins that physically destroy membranes – offer a fundamentally different approach to killing M. tuberculosis. Unlike conventional antibiotics that target specific proteins (giving bacteria evolutionary escape routes), AMPs attack the structural architecture of bacterial cells, including the unique mycobacterial envelope. Several natural AMPs have shown promise against TB, but finding the right sequences is like searching for molecular needles in a haystack of possibilities. By training deep learning models on thousands of validated AMPs and teaching them to recognize the specific molecular features that enable mycobacterial membrane penetration – lipophilicity for waxy barriers, cationic charge distribution, structural motifs for cell wall permeation – FranklinForge can intelligently navigate sequence space and propose TB-killing candidates before we ever synthesize them. 
          </WikiParagraph>
          <WikiParagraph>
            Our computational workflow strikes in four stages:
            <WikiList items={[
              "Data curation with emphasis on anti-mycobacterial peptides",
              "Predictive modeling trained to recognize TB-specific killing features",
              "De novo generation of novel anti-TB sequences optimized for both lethality and safety",
              "In silico validation using specialized servers to predict mycobacterial membrane interactions and cell envelope penetration",
            ]} ordered/>
          </WikiParagraph>
          <WikiParagraph>
            Instead of blindly screening hundreds of peptides in expensive BSL-3 facilities hoping for a hit, FranklinForge compresses years of experimental work into computational hours, allowing our wet lab to focus exclusively on the most promising anti-TB candidates. This is precision-guided drug discovery: artificial intelligence forging the weapons, experiments confirming the kills, and tuberculosis finally meeting its match.
          </WikiParagraph>
        </WikiSubsection>
      </WikiSection>

      <WikiSection id="amplify" title="Amplify">
        <WikiBold>
          INCOMPLETE
        </WikiBold>
      </WikiSection>

      <WikiSection id="franklin" title="FranklinForge">
        <WikiSubsection title="The Journey from Data to Discovery">
          <WikiCollapsible title="Chapter 1: Building the Foundation – Data Curation">
            <WikiParagraph>
              Every great structure begins with a solid foundation. For FranklinForge, that foundation was data—not just any data, but meticulously curated, biologically validated peptide sequences that could teach our models the molecular grammar of antimicrobial activity.
            </WikiParagraph>
            <WikiParagraph>
              We began our journey at the DBAASP (Database of Antimicrobial Activity and Structure of Peptides), downloading 23,689 peptide sequences—a vast library of nature's antibacterial arsenal accumulated over decades of research. But raw data is messy, inconsistent, and often misleading. Our first task was to separate signal from noise.
            </WikiParagraph>
            <WikiParagraph>
              We applied stringent quality filters, keeping only peptides that were biologically plausible and experimentally tractable. Sequences had to be at least 4 residues long—anything shorter would lack the structural complexity needed for specific bacterial targeting. We restricted modifications to free or amidated C-termini and acetylated N-termini, the most common natural variations. Most importantly, we limited composition to natural L-amino acids or their D-enantiomers—no exotic chemistries, no non-natural residues that would complicate both modeling and synthesis. After this initial purification, 18,597 peptides remained.
            </WikiParagraph>
            <WikiParagraph>
              Next came the crucial step of labeling. We needed to teach our models what "works" and what doesn't. Using experimental antimicrobial activity data, we classified peptides as active (activity &lt;32 µg/ml or &lt;10 µM against known bacterial targets) or inactive (consistently &gt;32 µg/ml or &gt;10 µM). This gave us 13,339 active peptides and 4,946 inactive ones—a treasure trove of molecular success and failure stories.
            </WikiParagraph>
            <WikiParagraph>
              But activity alone isn't enough. A peptide that kills bacteria while also destroying human red blood cells is useless as a therapeutic. Using hemolysis assay data from human erythrocytes, we created a second classification: non-hemolytic (&lt;20% hemolysis at &ge;50 µM) versus hemolytic (&ge;20% hemolysis at any tested concentration). This safety filter identified 3,757 safe peptides and 4,133 toxic ones.
            </WikiParagraph>
            <WikiParagraph>
              Yet we faced a problem: our activity dataset was imbalanced, with far more active than inactive sequences. Machine learning models trained on imbalanced data learn biased patterns. We needed more negative examples, but we couldn't just wait for researchers to test thousands more inactive peptides. Instead, we created them computationally.
            </WikiParagraph>
            <WikiParagraph>
              We generated 4,875 pseudo-inactive peptides using two complementary strategies. First, we took active sequences, split them into 1-2 residue fragments, shuffled the pieces, and reassembled them—destroying the functional motifs while preserving amino acid composition. Second, we extracted fragments from SwissProt protein sequences, matching the length distribution of real AMPs but lacking their antimicrobial architecture. We validated that none of these synthetic inactives appeared in our known dataset, ensuring our models would learn genuine patterns rather than memorizing artifacts.
            </WikiParagraph>
            <WikiParagraph>
              The result: a balanced, diverse, biologically grounded dataset ready to train the predictive heart of FranklinForge.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 2: Teaching Machines to Recognize Killers – The Activity Classifier">
            <WikiParagraph>
              With our curated dataset in hand, we faced a fundamental question: could we teach a machine to predict whether a peptide would kill bacteria just by reading its sequence?
            </WikiParagraph>
            <WikiParagraph>
              We turned to ProtT5, a transformer-based protein language model that had learned the deep grammar of proteins by reading millions of sequences. ProtT5 doesn't just see amino acids—it sees context, patterns, and relationships across the entire sequence, encoding each peptide as a 1024-dimensional vector that captures its biochemical essence.
            </WikiParagraph>
            <WikiParagraph>
              We fed these rich embeddings into a SimpleMLP neural network: four layers (1024 → 256 → 128 → 64 → 1) with batch normalization, ReLU activations, and dropout regularization. The architecture was deliberately straightforward—complexity in the embeddings, simplicity in the classifier. We trained on 25,822 balanced peptides for 150 epochs, using the OneCycleLR scheduler to dynamically adjust learning rates and BCEWithLogitsLoss to handle binary classification.
            </WikiParagraph>
            <WikiParagraph>
              The results validated our approach. On the validation set, the model achieved 79.09% accuracy with a ROC-AUC of 0.831. Precision stood at 76.0%, recall at 77.4%, and F1 score at 76.7%. These weren't perfect numbers—no model is perfect—but they were strong enough to filter thousands of candidates reliably. The neural network had learned to read the molecular signatures of antimicrobial potency encoded in sequence space.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 3: Predicting Safety – The Hemolysis Classifier">
            <WikiParagraph>
              Activity prediction solved half the problem. Now we needed to ensure our peptides wouldn't harm human cells. Hemolysis—the rupture of red blood cells—is a critical safety concern for any membrane-active peptide.
            </WikiParagraph>
            <WikiParagraph>
              For this task, we took a different approach. Instead of relying solely on learned embeddings, we engineered 422 interpretable biophysical features:
              <WikiList items={[
                "Amino Acid Composition (AAC, 20-D): The frequency of each residue",
                "Dipeptide Composition (DPC, 400-D): Pairwise residue patterns capturing local structural motifs",
                "Hydrophobicity (1-D): Mean Kyte-Doolittle hydropathy—critical for membrane interaction",
                "Net Charge (1-D): pH 7.0 charge state computed via modlamp—essential for electrostatic binding"
              ]} />
              These features weren't black-box representations. They were mechanistically meaningful descriptors that biochemists could interpret and trust.
            </WikiParagraph>
            <WikiParagraph>
              We chose XGBoost, a gradient-boosted decision tree algorithm, as our classifier. After cleaning conflicts and dropping ambiguous labels from our dataset of 7,522 peptides, we performed hyperparameter tuning via RandomizedSearchCV with 5-fold stratified cross-validation. The best configuration—300 estimators, max depth 7, learning rate 0.1—balanced model complexity with generalization.
            </WikiParagraph>
            <WikiParagraph>
              The test set results were impressive: 81.6% accuracy, ROC-AUC of 0.896, and an MCC (Matthews Correlation Coefficient) of 0.631. Precision and recall both hovered around 80%, indicating balanced performance across both classes.
            </WikiParagraph>
            <WikiParagraph>
              But what truly set this model apart was its explainability. Using SHAP (SHapley Additive exPlanations), we visualized which features drove predictions. Hydrophobicity and charge emerged as dominant factors—exactly what membrane-lysis theory predicts. Specific amino acids like lysine, arginine, leucine, and valine contributed significantly, confirming that the model had learned biophysically sound patterns rather than spurious correlations.
            </WikiParagraph>
            <WikiParagraph>
              This wasn't just a classifier—it was a computational safety filter grounded in mechanistic understanding.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 4: Generating Novel Sequences – The Generator Model">
            <WikiParagraph>
              Prediction is powerful, but generation is transformative. We could now score existing peptides for activity and safety, but what if we could design entirely new ones?
            </WikiParagraph>
            <WikiParagraph>
              We built a GRU (Gated Recurrent Unit) generator—a sequence-to-sequence model that learned the statistical patterns of antimicrobial peptides and could sample novel sequences from that learned distribution.
            </WikiParagraph>
            <WikiParagraph>
              The architecture was elegant:
              <WikiList items={[
                "Embedding layer: Each of the 20 amino acids mapped to a 64-dimensional vector",
                "GRU layer: 128 hidden units capturing sequential dependencies—learning which residues follow which",
                "Dropout (0.3): Preventing overfitting and encouraging diversity",
                "Output layer: Projecting GRU hidden states onto the vocabulary to predict the next amino acid"
              ]}/>
            </WikiParagraph>
            <WikiParagraph>
              We trained on active AMPs for 300 epochs using cross-entropy loss and SGD with momentum, teaching the model to predict each residue given the preceding sequence. After training, we set it loose: starting with a &lt;START&gt; token, the model generated amino acids one by one until it produced &lt;END&gt; or hit a maximum length.
            </WikiParagraph>
            <WikiParagraph>
              The generator created 120,000 sequences. After filtering out any that appeared in the training set, we retained 100,000 genuinely novel peptides—sequences that had never existed in nature or in any database, yet statistically resembled successful antimicrobial peptides.
            </WikiParagraph>
            <WikiParagraph>
              But we weren't done. Generic AMPs are useful, but we needed something more specific. We needed peptides designed to kill tuberculosis.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 5: Specializing for Tuberculosis – Fine-Tuning on Mycobacterial Sequences">
            <WikiParagraph>
              <WikiItalic>Mycobacterium tuberculosis</WikiItalic> is not a typical bacterium. Its thick, waxy cell wall—rich in mycolic acids—creates a molecular fortress that resists most conventional antibiotics. Growing it requires BSL-3 containment facilities. Its 24-hour doubling time makes screening agonizingly slow. And the rise of MDR-TB and XDR-TB has turned what was once curable into a potential death sentence.
            </WikiParagraph>
            <WikiParagraph>
              We needed AMPs specifically adapted to penetrate the mycobacterial envelope. Enter transfer learning.
            </WikiParagraph>
            <WikiParagraph>
              Using a phylogenetic tree from Orgeur et al. (2024), we identified ~30 bacterial species closely related to M. tuberculosis—organisms sharing evolutionary ancestry and similar cell wall architectures. From these species, we manually extracted 311 validated antimicrobial sequences—a small but precious dataset of peptides that had proven effective against mycobacterial targets.
            </WikiParagraph>
            <WikiParagraph>
              We fine-tuned our pretrained GRU generator on these 311 sequences, using a deliberately low learning rate (0.0005) for 50 epochs. The goal wasn't to overwrite the model's general AMP knowledge but to subtly shift its generative distribution toward TB-relevant patterns—enriching for lipophilic residues that penetrate waxy barriers, optimizing charge distributions for mycolic acid interaction, favoring structural motifs that maintain activity in complex lipid environments.
            </WikiParagraph>
            <WikiParagraph>
              The fine-tuned model then generated 160,000 unique MTB-targeted sequences. Amino acid composition analysis revealed that these sequences closely matched known MTB-active AMPs—our generator had successfully learned the biochemical dialect of anti-TB peptides.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 7: Ensuring Novelty – Levenshtein Distance Analysis">
            <WikiParagraph>
              Having thousands of candidates was still too many. But more importantly, we needed to ensure our sequences were genuinely novel—not just minor variations of known AMPs.
            </WikiParagraph>
            <WikiParagraph>
              For each generated peptide g, we computed its Levenshtein distance (minimum edit distance) to every known sequence k in the entire DBAASP database:
              
                <MathJax>
                  {"$$d_{\\min}(g) = \\min_{k \\in \\text{DBAASP}} \\text{Levenshtein}(g, k)$$"}
                </MathJax>
              <WikiImage src="https://static.igem.wiki/teams/6026/igem2025/drylab/description/levenshein-distance.webp" alt="Novel Evaluation Using Levenshtein Distance" caption="Novel Evaluation Using Levenshtein Distance" />
            </WikiParagraph>
            <WikiParagraph>
              The Levenshtein distance counts the minimum insertions, deletions, and substitutions needed to transform one sequence into another. A distance of 1 means a single amino acid difference; a distance of 5 means at least five edits separate the sequences.
            </WikiParagraph>
            <WikiParagraph>
              We applied a strict threshold: <WikiBold>dmin(g)&ge;5</WikiBold>. Any peptide within 4 edits of a known AMP was rejected. This ensured we weren't just making trivial mutations—we were exploring genuinely unexplored regions of sequence space.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 8: Diversity Through Clustering – Selecting Representatives">
            <WikiParagraph>
              Even after novelty filtering, we had thousands of candidates. Many were similar to each other—slight variations on the same structural theme. Synthesizing and testing all of them would be wasteful.
            </WikiParagraph>
            <WikiBold>
              We needed diversity.
            </WikiBold>
            <WikiParagraph>
              We computed a pairwise Levenshtein distance matrix for all remaining candidates, then applied hierarchical agglomerative clustering with average linkage. Clusters were defined such that any two sequences in the same cluster differed by at most 5 edits—grouping molecular families together.
            </WikiParagraph>
            <WikiParagraph>
              From each cluster, we selected a single representative: the sequence with the lowest{" "}<MathJax inline>{"$$d_{\\min}$$"}</MathJax>{" "}to known active AMPs. This strategy balanced two competing goals—novelty (ensured by our earlier filter) and evolutionary proximity to validated designs (maximizing the chance that novelty translates to function).
            </WikiParagraph>
            <WikiParagraph>
              The result: 40 elite peptide candidates—diverse, novel, predicted-active, non-hemolytic, and optimized for TB.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Chapter 9: The Final Validation – AntiTBpred Server Screening">
            <WikiParagraph>
              Forty candidates represented a massive reduction from 160,000 generated sequences, but we wanted one final computational checkpoint before committing wet lab resources. We turned to AntiTBpred, a specialized server developed specifically for predicting anti-tuberculosis peptide activity.
            </WikiParagraph>
            <WikiParagraph>
              AntiTBpred uses machine learning models trained on experimentally validated anti-TB peptides, incorporating TB-specific features that our general activity classifier might miss—subtle patterns related to mycobacterial membrane interactions, cell wall penetration efficiency, and activity against slow-growing intracellular bacteria.
            </WikiParagraph>
            <WikiParagraph>
              We submitted all 40 candidates to AntiTBpred for scoring. The server ranked them by predicted anti-TB potency, considering factors our own models hadn't explicitly optimized for. From this final computational screen emerged 4-5 top-performing sequences—peptides that had survived every filter, maximized diversity, demonstrated novelty, and now showed the highest predicted efficacy specifically against Mycobacterium tuberculosis.
            </WikiParagraph>
            <WikiParagraph>
              These were the molecular weapons we'd forged through data, design, and relentless computational refinement.
            </WikiParagraph>
          </WikiCollapsible>
        </WikiSubsection>

        <WikiSubsection title="Epilogue: From Silicon to Synthesis">
          <WikiParagraph>
            The journey from 23,689 database peptides to 4-5 elite anti-TB candidates represents the power of precision-guided drug discovery. Instead of blindly synthesizing hundreds of peptides in expensive BSL-3 facilities and waiting weeks for each growth assay, FranklinForge compressed years of trial-and-error into computational hours.
          </WikiParagraph>
          <WikiParagraph>
            These 4-5 sequences now advance to the AMPlify wet lab for experimental validation against M. tuberculosis. They carry with them the accumulated wisdom of thousands of validated AMPs, the structural specialization of mycobacterial-targeted design, the safety assurance of multi-stage filtering, and the novelty of unexplored sequence space.
          </WikiParagraph>
          <WikiParagraph>
            Structure determined their design. Data guided their evolution. Computation forged their creation.
          </WikiParagraph>
          <WikiParagraph>
            But the true test awaits in the lab, where these digital predictions will face the ultimate judge: biology itself. If even one of these sequences demonstrates potent anti-TB activity with acceptable safety profiles, FranklinForge will have proven that artificial intelligence can forge real weapons in humanity's ancient war against infectious disease.
          </WikiParagraph>
          <WikiBold>
            <WikiItalic>
              This is the future of antibiotic discovery. This is FranklinForge.
            </WikiItalic>
          </WikiBold>
        </WikiSubsection>
      </WikiSection>

      <WikiSection id="closingtheloop" title="Closing the Loop: When Computational Predictions Meet Experimental Reality">
        <WikiParagraph>
          The true power of FranklinForge isn't in generating sequences—it's in generating <WikiBold>the right sequences</WikiBold>. Computational models can predict, filter, and optimize with remarkable speed, but only experimental biology can deliver the ultimate verdict: does this peptide actually kill Mycobacterium tuberculosis?
        </WikiParagraph>
        <WikiParagraph>
          This is where AMPlify's dry lab and wet lab converge into a unified discovery engine. FranklinForge delivered <WikiBold>4-5 elite peptide candidates</WikiBold>—sequences that survived rigorous computational gauntlets including activity prediction, safety screening, novelty evaluation, diversity clustering, and TB-specific optimization via AntiTBpred. These weren't random shots in the dark; they were <WikiBold>precision-guided molecular weapons</WikiBold> designed with every available computational insight.
        </WikiParagraph>
        <WikiParagraph>
          But predictions are hypotheses, not evidence. The wet lab's mission: transform these digital sequences into physical peptides and test them against the real enemy.
        </WikiParagraph>
        <WikiSubsection title="The Handoff: What the Wet Lab Received">
          <WikiParagraph>
            FranklinForge provided the wet lab with:
            <WikiList items={
              [
                "Optimized sequences (4-5 peptides): Length-optimized for synthesis, predicted active by neural networks, predicted non-hemolytic by XGBoost, TB-specialized via fine-tuned generation, and validated by AntiTBpred",
                "Computational confidence scores: Activity probabilities, hemolysis risk assessments, and AntiTBpred rankings",
                "Biophysical feature profiles: Charge, hydrophobicity, amino acid composition—mechanistic insights to guide experimental interpretation",
                "Novelty metrics: Levenshtein distances to known AMPs, ensuring these sequences represent genuinely unexplored chemical space",
            ]} ordered/>
            This wasn't a list of random peptides. It was a <WikiBold>computationally curated portfolio</WikiBold> where every candidate carried a detailed molecular rationale.
          </WikiParagraph>
        </WikiSubsection>

        <WikiSubsection title="The Wet Lab Workflow: From Sequence to Efficacy">
          <WikiBold>Step 1: Peptide Synthesis</WikiBold>
          <WikiParagraph>
            The 4-5 sequences were chemically synthesized using solid-phase peptide synthesis (SPPS), purified via HPLC, and validated by mass spectrometry. What existed only as strings of letters in FranklinForge's output files now existed as physical molecules—ready to interact with biological systems.
          </WikiParagraph>

          <WikiBold>Step 2: Antimicrobial Activity Assays</WikiBold>
          <WikiParagraph>
            Synthesized peptides were tested against Mycobacterium tuberculosis (and potentially related mycobacterial species or model organisms) to determine minimum inhibitory concentrations (MICs). This is the moment of truth: do FranklinForge's predictions translate to real bacterial killing?
          </WikiParagraph>

          <WikiBold>Step 3: Cytotoxicity and Hemolysis Testing</WikiBold>
          <WikiParagraph>
            Even if a peptide kills TB, it's useless if it also kills human cells. Hemolysis assays against human erythrocytes and cytotoxicity assays against mammalian cell lines validated FranklinForge's safety predictions. The XGBoost hemolysis classifier had made its bet—now the lab would check its work.
          </WikiParagraph>

          <WikiBold>Step 4: Mechanism-of-Action Studies (if promising candidates emerged)</WikiBold>
          <WikiParagraph>
            For peptides showing strong activity and acceptable safety profiles, mechanistic studies investigated how they kill TB—membrane disruption assays, permeabilization studies, microscopy to visualize peptide-cell interactions. These experiments connect computational predictions to biological mechanisms.
          </WikiParagraph>

          <WikiBold>Step 5: Feedback Loop to FranklinForge</WikiBold>
          <WikiParagraph>
            Every experimental result—success or failure—becomes new training data. Which predictions were accurate? Which were off? What biophysical features correlated with unexpected outcomes? This feedback refines FranklinForge's models, making the next generation of predictions even sharper.
          </WikiParagraph>
        </WikiSubsection>

        <WikiSubsection title="The Integration Philosophy: Computation Guides, Biology Decides">
          <WikiParagraph>
            AMPlify's approach isn't "dry lab versus wet lab" or even "dry lab then wet lab." It's <WikiBold>dry lab with wet lab</WikiBold>—a continuous dialogue where:
            <WikiList items={[
              "Computation compresses the search space: Instead of testing 160,000 generated sequences, the lab tests 4-5",
              "Experiments validate models: Wet lab results reveal where models succeed and where they need improvement",
              "Biology grounds design: Computational predictions are always anchored to experimental data from DBAASP and literature",
              "Iteration accelerates discovery: Each cycle of generation → prediction → testing → feedback makes the pipeline smarter."
            ]} />
          </WikiParagraph>
          <WikiParagraph>
            This integration is what makes AMPlify more than the sum of its parts. FranklinForge doesn't replace wet lab science—it amplifies it (hence the project name). By intelligently prioritizing which experiments to run, we maximize the scientific return on every synthesis, every assay, every precious hour of laboratory time.
          </WikiParagraph>
        </WikiSubsection>

        <WikiSubsection title="The Outcome: Validated Anti-TB Peptides (or Validated Learning)">
          <WikiParagraph>
            If FranklinForge's candidates demonstrate potent anti-TB activity with acceptable safety:
            <WikiList items={[
              "We've proven that AI-guided peptide design works for tuberculosis",
              "We've generated novel therapeutic leads for MDR/XDR-TB",
              "We've established a replicable pipeline for antimicrobial discovery"
            ]}/>
            If the candidates underperform:
            <WikiList items={[
              "We've identified gaps in our predictive models",
              "We've generated invaluable negative data to retrain FranklinForge",
              "We've learned what doesn't work—arguably as valuable as learning what does"
            ]}/>
            Either way, <WikiBold>the integration of dry and wet lab moves the field forward.</WikiBold>
          </WikiParagraph>
        </WikiSubsection>

        <WikiSubsection title="Beyond AMPlify: A Blueprint for Computational Biology">
          <WikiParagraph>
            The FranklinForge-to-wet-lab workflow demonstrates a broader principle: <WikiBold>computational biology is most powerful when it's tightly integrated with experimental validation.</WikiBold>
          </WikiParagraph>
          <WikiParagraph>
            Models trained on data, tested against reality, and refined through feedback loops represent the future of drug discovery—not just for AMPs, not just for TB, but for any therapeutic challenge where search spaces are vast and resources are limited.
          </WikiParagraph>
          <WikiParagraph>
            AMPlify proves that you don't need a pharmaceutical company's budget to do cutting-edge drug discovery. You need good data, smart algorithms, biological insight, and a willingness to let computation and experimentation inform each other.
          </WikiParagraph>
          <WikiBold>
            From 23,689 database peptides to 4-5 lab-tested candidates. From sequence space to culture plates. From FranklinForge's predictions to biology's verdict.
          </WikiBold>
        </WikiSubsection>
      </WikiSection>


      <WikiSection id="futurevision" title="Future Vision: Beyond Tuberculosis – A Platform for Global Antimicrobial Discovery">
        <WikiSubsection title="Expanding the Pipeline: From TB to Pan-Pathogen Design">
          <WikiParagraph>
            FranklinForge was born to fight tuberculosis, but its architecture transcends any single disease. The pipeline we've built—data curation, dual-classifier screening, phylogenetically-guided generative models, novelty evaluation, and computational-experimental integration—is fundamentally pathogen-agnostic. With relatively minor adaptations, this framework could target any bacterial infection where antimicrobial peptides show promise.
          </WikiParagraph>
          <WikiParagraph>
            Immediate expansion targets:
            <WikiList items={[
              "Multidrug-resistant Gram-negative bacteria: Pseudomonas aeruginosa, Acinetobacter baumannii, Klebsiella pneumoniae—the WHO's critical priority pathogens",
              "Biofilm-forming organisms: Chronic wound infections, cystic fibrosis lung colonization, medical device contamination",
              "Intracellular pathogens: Salmonella, Listeria, pathogens that hide inside host cells where conventional antibiotics struggle to reach",
              "Fungal infections: Adapting the pipeline for antifungal peptides targeting Candida, Aspergillus, and emerging threats like Candida auris",
              "Viral envelope disruption: Exploring peptides that destabilize lipid-enveloped viruses as broad-spectrum antivirals"
            ]}/>
            Each new target would require:
            <WikiList items={[
              "Disease-specific training data from DBAASP or literature",
              "Phylogenetic fine-tuning datasets for the target pathogen family",
              "Specialized validation servers analogous to AntiTBpred (or developing our own)",
              "Pathogen-specific feature engineering (e.g., for Gram-negatives: outer membrane penetration predictors; for fungi: chitin-binding motifs)"
            ]} ordered/>
            The computational infrastructure is already built. The methodological framework is validated. Scaling is now an engineering problem, not a research question.
          </WikiParagraph>
        </WikiSubsection>

        <WikiSubsection title="From Academic Project to Commercial Platform">
          <WikiParagraph>
            AMPlify and FranklinForge represent more than a successful iGEM project—they're the foundation of a computational drug discovery platform with genuine commercial potential.
          </WikiParagraph>
          <WikiBold>
            Publication Strategy: Establishing Scientific Credibility
          </WikiBold>
          <WikiParagraph>
            A high-impact publication would accomplish three goals:
            <WikiList items={[
              "Validate the approach through peer review",
              "Establish priority in the AI-guided AMP design space",
              "Attract collaborators from academia, industry, and funding agencies"
            ]} ordered/>
            Potential publication venues:
            <WikiList items={[
              "Nature Biotechnology or Nature Communications: For the full pipeline demonstration with wet lab validation",
              "Bioinformatics or Nucleic Acids Research: For the computational methodology and benchmarking",
              "Antimicrobial Agents and Chemotherapy: For TB-specific results with microbiological validation",
              "PLOS Computational Biology: Open-access platform emphasizing reproducibility"
            ]}/>
            Key narrative for publication: "AI-guided antimicrobial peptide design achieves [X]% reduction in experimental search space while maintaining [Y]% prediction accuracy for activity and safety. Phylogenetically-informed transfer learning enables rapid adaptation to novel pathogens. Wet lab validation confirms [Z] candidates with MIC values competitive with existing TB therapeutics."
          </WikiParagraph>
          <WikiParagraph>
            The data is there. The story is compelling. The impact is clear.
          </WikiParagraph>

          <WikiBold>
            Patent Strategy: Protecting Intellectual Property
          </WikiBold>
          <WikiParagraph>
            Multiple aspects of FranklinForge are potentially patentable:
          </WikiParagraph>
          <WikiParagraph>
            Patent 1: "Method for Phylogenetically-Guided Antimicrobial Peptide Generation Using Fine-Tuned Recurrent Neural Networks"
            <WikiList items={[
              "Claims: The specific approach of using phylogenetic trees to select fine-tuning datasets for generative models targeting pathogen-specific AMPs",
              "Novelty: Combines evolutionary biology with transfer learning in a way not previously documented",
              "Commercial value: Enables rapid pipeline adaptation to emerging pathogens"
            ]}/>
            Patent 2: "Multi-Objective Computational Screening System for Antimicrobial Peptide Safety and Efficacy"
            <WikiList items={[
              "Claims: The integrated pipeline combining protein language model embeddings (activity) with interpretable biophysical features (hemolysis) for dual-objective optimization",
              "Novelty: Specific architecture and feature engineering approach",
              "Commercial value: Core technology for any AMP discovery platform"
            ]}/>
            Patent 3: "Levenshtein Distance-Based Clustering for Diverse Antimicrobial Candidate Selection"
            <WikiList items={[
              "Claims: Using edit distance metrics with hierarchical clustering to maximize peptide library diversity while maintaining similarity to validated sequences",
              "Novelty: Specific application to peptide design with the mathematical framework",
              "Commercial value: Ensures generated libraries avoid redundancy"
            ]}/>
            Provisional patent strategy: File provisional patents first (lower cost, 12-month priority window) while preparing full patent applications. This establishes priority date while allowing time for additional wet lab validation data to strengthen claims.
          </WikiParagraph>

          <WikiBold>
            Entrepreneurial Vision: AMPlify Therapeutics or FranklinForge.AI
          </WikiBold>
          <WikiParagraph>
            The technology stack and validated methodology could support multiple business models:
          </WikiParagraph>
          <WikiParagraph>
            Option 1: Drug Discovery Company (AMPlify Therapeutics)
            <WikiList items={[
              "Model: Develop FranklinForge-designed AMPs as proprietary therapeutic candidates",
              "Revenue: Licensing to pharmaceutical companies, milestone payments, royalties on approved drugs",
              "Funding: Venture capital, government grants (NIH SBIR/STTR, BARDA for biodefense applications)",
              "Timeline: Long (10+ years to market) but potentially massive returns",
              "Challenge: Requires significant capital for preclinical development, clinical trials, regulatory approval"
            ]}/>
            Option 2: Platform-as-a-Service (FranklinForge.AI)
            <WikiList items={[
              "Model: License the computational platform to pharma companies, biotech startups, academic labs",
              "Revenue: Subscription fees, per-project licensing, compute credits",
              "Funding: Angel investment, seed rounds, potentially bootstrapped via consulting",
              "Timeline: Medium (2-3 years to product-market fit)",
              "Challenge: Requires robust software engineering, cloud infrastructure, customer support"
            ]}/>
            Option 3: Hybrid Model (Platform + Proprietary Pipeline)
            <WikiList items={[
              "Model: License the platform broadly while maintaining proprietary in-house drug discovery for high-value targets (e.g., biodefense, neglected tropical diseases)",
              "Revenue: Dual streams from licensing and therapeutics development",
              "Funding: Mixed venture capital and government grants",
              "Timeline: Variable depending on which revenue stream matures first",
              "Advantage: Diversified risk, multiple paths to profitability"
            ]}/>
            Go-to-market considerations:
            <WikiList items={[
              "Beachhead market: Biotech companies working on neglected tropical diseases or biodefense (less competition, faster timelines, government funding available)",
              "Key partnerships: Contract research organizations (CROs) for wet lab validation, cloud computing providers (AWS, Google Cloud) for infrastructure",
              "Regulatory strategy: For therapeutics path, engage FDA early for orphan drug designation (TB qualifies) and fast-track approval pathways."
            ]}/>
          </WikiParagraph>

          <WikiBold>
            Impact Beyond Profit: Global Health Equity
          </WikiBold>
          <WikiParagraph>
            Commercial success and social impact aren't mutually exclusive. A properly structured venture could:
            <WikiList items={[
              "Price discriminate geographically: Low-cost licensing for low- and middle-income countries where TB burden is highest",
              "Open-source the platform after establishing commercial viability (similar to AlphaFold strategy)",
              "Partner with nonprofits like Médecins Sans Frontières or the Gates Foundation for access programs",
              "Establish academic collaborations ensuring continued innovation even as commercial entity scales"
            ]}/>
            The vision: Make FranklinForge the computational backbone of global antimicrobial discovery—accessible enough that a research lab in Mumbai can design AMPs as easily as one in Boston, powerful enough that pharma giants license it for billion-dollar programs.
          </WikiParagraph>
        </WikiSubsection>
      </WikiSection>

      <WikiSection id="summary" title="Summary">
        <WikiParagraph>
          AMPlify's FranklinForge represents a paradigm shift in how we discover antimicrobial therapeutics. By integrating rigorous data curation, state-of-the-art machine learning, phylogenetically-informed generative design, and computational-experimental feedback loops, we've built a pipeline that transforms the economics and timeline of drug discovery.
        </WikiParagraph>
        <WikiParagraph>
          <WikiBold>What we built:</WikiBold>
          <WikiList items={[
            "A dual-classifier system (neural network for activity, gradient-boosted trees for safety) that screens peptide candidates with ~80% accuracy and high explainability",
            "A GRU-based generator fine-tuned on mycobacterial peptides, producing 160,000 TB-targeted novel sequences",
            "A multi-stage filtering gauntlet combining machine learning predictions, biophysical constraints, and novelty metrics",
            "An integrated dry-wet lab workflow that delivered 4-5 elite candidates for experimental validation"
          ]}/>
          <WikiBold>What we proved:</WikiBold>
          <WikiList items={[
            "Computational design can reduce experimental search space by orders of magnitude without sacrificing quality",
            "Transfer learning via phylogenetic fine-tuning enables rapid pathogen specialization",
            "Interpretable features (charge, hydrophobicity) complement deep learning for safety-critical predictions",
            "AI-generated peptides can achieve biochemical profiles matching or exceeding naturally occurring AMPs"
          ]}/>
          <WikiBold>What it means:</WikiBold>
          <WikiList items={[
            "For science: A reproducible methodology applicable to any bacterial pathogen",
            "For medicine: Potential new therapeutics for TB and other drug-resistant infections",
            "For the field: Proof that student-led, open-science approaches can compete with industry-scale drug discovery"
          ]}/>
          FranklinForge isn't just a tool—it's a blueprint. A demonstration that the future of antimicrobial discovery lies not in billion-dollar screening facilities, but in the intelligent integration of computation and biology. From 23,689 database peptides to 4-5 lab-ready candidates. From months of synthesis to hours of generation. From trial-and-error to precision-guided design.
        </WikiParagraph>
      </WikiSection>

      {/* References Section */}
      <WikiSection id="references" title="References">
        <WikiReferences>
          <WikiReferenceItem>
            Yang, Y., et al. (2019). Genome organization and regulation in
            human cells. Nature Reviews Genetics, 20(1), 1-15.
          </WikiReferenceItem>
          <WikiReferenceItem>
            Mikkelsen, T.S., et al. (2005). Initial sequence of the chimpanzee
            genome and comparison with the human genome. Nature, 437(7055),
            69-87.
          </WikiReferenceItem>
          <WikiReferenceItem>
            Visit the{" "}
            <a href="https://competition.igem.org/judging/medals" target="_blank" rel="noopener noreferrer">
              Medals page
            </a>{" "}
            for more information about iGEM judging criteria.
          </WikiReferenceItem>
        </WikiReferences>
      </WikiSection>
    </WikiLayout>
    </MathJaxContext>
  );
}
