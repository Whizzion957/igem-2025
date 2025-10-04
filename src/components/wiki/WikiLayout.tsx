import React, { useEffect, useState, useRef } from "react";
import "./wiki-layout.css";

interface Section {
  id: string;
  title: string;
}

interface WikiLayoutProps {
  title: string;
  sections: Section[];
  children: React.ReactNode;
}

export function WikiLayout({ title, sections, children }: WikiLayoutProps) {
  const [activeSection, setActiveSection] = useState<string>("");
  const [scrollProgress, setScrollProgress] = useState(0);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      // Calculate scroll progress
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);

      // Find active section
      const sectionElements = sections.map(s => document.getElementById(s.id));
      for (let i = sectionElements.length - 1; i >= 0; i--) {
        const element = sectionElements[i];
        if (element && element.getBoundingClientRect().top <= 150) {
          setActiveSection(sections[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [sections]);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="wiki-page-container">
      {/* Progress Bar */}
      <div className="wiki-progress-bar" style={{ height: `${scrollProgress}%` }} />

      {/* Sidebar TOC */}
      <aside className="wiki-sidebar">
        <h2 className="wiki-sidebar-title">{title}</h2>
        <nav className="wiki-toc-nav">
          <ul>
            {sections.map((section) => (
              <li key={section.id}>
                <button
                  className={`wiki-toc-link ${activeSection === section.id ? "active" : ""}`}
                  onClick={() => scrollToSection(section.id)}
                >
                  {section.title}
                </button>
              </li>
            ))}
            <li>
              <button
                className="wiki-back-to-top"
                onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
              >
                Back to the top
              </button>
            </li>
          </ul>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="wiki-main-content" ref={contentRef}>
        {children}
      </main>
    </div>
  );
}

// Reusable Content Components

interface WikiSectionProps {
  id: string;
  title: string;
  children: React.ReactNode;
}

export function WikiSection({ id, title, children }: WikiSectionProps) {
  return (
    <section className="wiki-content-section" id={id}>
      <h2 className="wiki-section-heading">{title}</h2>
      {children}
    </section>
  );
}

interface WikiSubsectionProps {
  title: string;
  children: React.ReactNode;
}

export function WikiSubsection({ title, children }: WikiSubsectionProps) {
  return (
    <div className="wiki-subsection">
      <h3 className="wiki-subsection-heading">{title}</h3>
      {children}
    </div>
  );
}

interface WikiSummaryCardProps {
  title: string;
  icon?: string;
  children: React.ReactNode;
}

export function WikiSummaryCard({ title, icon, children }: WikiSummaryCardProps) {
  return (
    <div className="wiki-summary-card">
      {icon && <div className="wiki-card-icon">{icon}</div>}
      <h3 className="wiki-card-title">{title}</h3>
      <div className="wiki-card-content">{children}</div>
    </div>
  );
}

interface WikiImageProps {
  src: string;
  alt: string;
  caption?: string;
}

export function WikiImage({ src, alt, caption }: WikiImageProps) {
  return (
    <figure className="wiki-image-figure">
      <img src={src} alt={alt} className="wiki-content-image" />
      {caption && <figcaption className="wiki-image-caption">{caption}</figcaption>}
    </figure>
  );
}

interface WikiCollapsibleProps {
  title: string;
  children: React.ReactNode;
}

export function WikiCollapsible({ title, children }: WikiCollapsibleProps) {
  return (
    <details className="wiki-collapsible">
      <summary className="wiki-collapsible-summary">{title}</summary>
      <div className="wiki-collapsible-content">{children}</div>
    </details>
  );
}

interface WikiListProps {
  items: (string | React.ReactNode)[];
  ordered?: boolean;
}

export function WikiList({ items, ordered = false }: WikiListProps) {
  const ListTag = ordered ? "ol" : "ul";
  return (
    <ListTag className="wiki-content-list">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ListTag>
  );
}

export function WikiParagraph({ children }: { children: React.ReactNode }) {
  return <p className="wiki-paragraph">{children}</p>;
}

export function WikiBold({ children }: { children: React.ReactNode }) {
  return <strong className="wiki-bold">{children}</strong>;
}

export function WikiItalic({ children }: { children: React.ReactNode }) {
  return <em className="wiki-italic">{children}</em>;
}

export function WikiUnderline({ children }: { children: React.ReactNode }) {
  return <span className="wiki-underline">{children}</span>;
}

export function WikiReferences({ children }: { children: React.ReactNode }) {
  return (
    <div className="wiki-references-section">
      <h3>References</h3>
      <div className="wiki-references-content">{children}</div>
    </div>
  );
}

export function WikiReferenceItem({ children }: { children: React.ReactNode }) {
  return <div className="wiki-reference-item">{children}</div>;
}
