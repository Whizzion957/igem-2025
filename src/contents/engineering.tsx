import {
  WikiLayout,
  WikiSection,
  WikiSubsection,
  WikiSummaryCard,
  WikiParagraph,
  WikiList,
  WikiBold,
  WikiCollapsible,
  WikiReferences,
  WikiReferenceItem,
} from "../components/wiki/WikiLayout";

export function Engineering() {
  const sections = [
    { id: "overview", title: "Engineering Overview" },
    { id: "design-cycle", title: "Design Cycle" },
    { id: "implementation", title: "Implementation" },
    { id: "tips", title: "Tips for Success" },
    { id: "references", title: "References" },
  ];

  return (
    <WikiLayout title="Engineering Success" sections={sections}>
      {/* Summary Card */}
      <WikiSummaryCard title="Silver Medal Criterion #1" icon="🏅">
        <WikiParagraph>
          Demonstrate engineering success in a technical aspect of your project
          by going through at least one iteration of the engineering design
          cycle: <WikiBold>Design → Build → Test → Learn</WikiBold>
        </WikiParagraph>
      </WikiSummaryCard>

      {/* Overview Section */}
      <WikiSection id="overview" title="Engineering Overview">
        <WikiParagraph>
          Engineering success can be achieved by documenting your effort to
          follow the engineering design cycle: Design → Build → Test → Learn
        </WikiParagraph>
        <WikiParagraph>
          We invite you to think about ways to tackle and solve one or more of
          your project's problems and use synthetic biology tools and/or
          experimental techniques to generate expected results.
        </WikiParagraph>
      </WikiSection>

      {/* Design Cycle Section */}
      <WikiSection id="design-cycle" title="Design Cycle">
        <WikiSubsection title="Core Assumptions">
          <WikiCollapsible title="Design Phase">
            <WikiParagraph>
              In the design phase, identify the problem you want to solve and
              conceptualize a solution. Consider the biological parts,
              systems, or approaches you'll use.
            </WikiParagraph>
            <WikiList
              items={[
                "Define clear project objectives",
                "Research existing solutions and approaches",
                "Select appropriate biological parts and methods",
                "Plan experimental protocols",
              ]}
            />
          </WikiCollapsible>

          <WikiCollapsible title="Build Phase">
            <WikiParagraph>
              Implement your design by constructing the biological system,
              assembling parts, or setting up experimental procedures.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Test Phase">
            <WikiParagraph>
              Evaluate your construction through rigorous testing and data
              collection to determine if it performs as expected.
            </WikiParagraph>
          </WikiCollapsible>

          <WikiCollapsible title="Learn Phase">
            <WikiParagraph>
              Analyze your results, identify what worked and what didn't, and
              use these insights to inform the next iteration of your design.
            </WikiParagraph>
          </WikiCollapsible>
        </WikiSubsection>
      </WikiSection>

      {/* Implementation Section */}
      <WikiSection id="implementation" title="Implementation">
        <WikiParagraph>
          When you have completed the cycle once, think about and document what
          changes in design you would make for the next iteration(s) of the
          cycle.
        </WikiParagraph>
        <WikiParagraph>
          For example, you can design and build a new Part, measure its
          performance, document whether it worked or not, and propose how the
          results would inform the next design or steps (documentation must be
          on the Part's Pages on the Registry).
        </WikiParagraph>
      </WikiSection>

      {/* Tips Section */}
      <WikiSection id="tips" title="Tips for Success">
        <WikiList
          items={[
            "Document each iteration of the design cycle thoroughly",
            "Be specific about what worked and what didn't work",
            "Explain how each iteration informed the next",
            "Include data, measurements, and quantitative results",
            "Show how you improved your design based on testing results",
            "Demonstrate multiple iterations when possible",
          ]}
        />
      </WikiSection>

      {/* References Section */}
      <WikiSection id="references" title="References">
        <WikiReferences>
          <WikiReferenceItem>
            Visit the{" "}
            <a href="https://competition.igem.org/judging/medals" target="_blank" rel="noopener noreferrer">
              Medals page
            </a>{" "}
            for more information about iGEM judging criteria.
          </WikiReferenceItem>
          <WikiReferenceItem>
            Visit the{" "}
            <a href="https://technology.igem.org/engineering" target="_blank" rel="noopener noreferrer">
              Engineering pages
            </a>{" "}
            for additional guidance on engineering success.
          </WikiReferenceItem>
        </WikiReferences>
      </WikiSection>
    </WikiLayout>
  );
}
