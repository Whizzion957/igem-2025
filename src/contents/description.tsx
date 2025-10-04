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
} from "../components/wiki/WikiLayout";

export function Description() {
  const sections = [
    { id: "abstract", title: "Abstract" },
    { id: "introduction", title: "Introduction" },
    { id: "project-overview", title: "Project Overview" },
    { id: "references", title: "References" },
  ];

  return (
    <WikiLayout title="Project Description" sections={sections}>
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
        <WikiParagraph>
          Cells interpret identical DNA sequences differently depending on the
          3D genome organization, leading to diverse cellular phenotypes and
          even determining between health and disease. While synthetic biology
          provides powerful methods to engineer the genome sequence, tools to
          sculpt the genome spatial architecture are lacking.
        </WikiParagraph>
        <WikiParagraph>
          We address this profound gap by introducing{" "}
          <WikiBold>PlCasS0</WikiBold>, a toolbox for rationally engineering
          the 3D genome organization. Using different CRISPR/Cas orthologs and
          new, engineered fusion guide RNAs, we created "Cas staples" that bind
          and connect otherwise non-interacting genomic regions.
        </WikiParagraph>
        <WikiParagraph>
          To streamline the development cycle of new Cas staples, we developed
          DaVinci, a multi-faceted model integrating both local and long-range
          DNA-protein interactions into a unified pipeline.
        </WikiParagraph>
        <WikiParagraph>
          Our comprehensive part collection and extensive education program
          enable the next generation of iGEMers to efficiently adapt PlCasS0
          and engineer genomes in 3D.
        </WikiParagraph>
      </WikiSection>

      {/* Introduction Section */}
      <WikiSection id="introduction" title="Introduction">
        <WikiParagraph>
          All cells in our body carry the same genetic sequence, yet they have
          highly diverging and specialized functions. Even more, species that
          look and behave profoundly differently like humans and chimpanzees
          share almost identical genome sequences (Yang et al., 2019; Mikkelsen
          et al., 2005).
        </WikiParagraph>
        <WikiSubsection title="Background Information">
          <WikiParagraph>
            The answer to this apparent contradiction lies, at least in parts,
            in another dimension used by living systems for information
            encoding: The 3D genome conformation. In fact, cells can interpret
            identical DNA sequences in various ways depending on their spatial
            genome organization.
          </WikiParagraph>
          <WikiList
            items={[
              "Understanding genome organization is crucial for synthetic biology",
              "3D structure affects gene expression and cellular function",
              "New tools are needed to engineer spatial genome architecture",
            ]}
          />
        </WikiSubsection>
      </WikiSection>

      {/* Project Overview Section */}
      <WikiSection id="project-overview" title="Project Overview">
        <WikiSubsection title="What Should this Page Contain?">
          <WikiList
            items={[
              "Explain the problem your project addresses and its potential impact",
              "Provide a clear and concise summary of your project's goals and objectives",
              "Detail the specific reasons why your team chose this project",
              "Explain the inspiration behind your project, including any prior research or real-world problems that motivated your team",
              "Use illustrations, diagrams, and other visual aids to enhance understanding",
              "Include relevant scientific background, technical details, and experimental approaches",
            ]}
          />
        </WikiSubsection>

        <WikiSubsection title="Tips for Success">
          <WikiList
            items={[
              "While providing detailed information, strive for clarity and conciseness",
              "Use summaries and subheadings to organize your content",
              "Utilize visuals to enhance understanding and engagement",
              "Document your research process and sources thoroughly",
            ]}
          />
        </WikiSubsection>
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
  );
}
