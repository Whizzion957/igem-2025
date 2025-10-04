import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import stage01 from "../assets/parallax/stage-01.svg";
import stage02 from "../assets/parallax/stage-02.svg";
import stage04 from "../assets/parallax/stage-04.svg";
import stage07 from "../assets/parallax/stage-07.svg";
import stage08 from "../assets/parallax/stage-08.svg";
import stage09 from "../assets/parallax/stage-09.svg";
import stage10 from "../assets/parallax/stage-10.svg";
import stage11 from "../assets/parallax/stage-11.svg";
import stage12 from "../assets/parallax/stage-12.svg";
import stage13 from "../assets/parallax/stage-13.svg";

gsap.registerPlugin(ScrollTrigger);

const stages = [
  { id: 1, src: stage01, label: "AMPify: Fighting Bacterial Infections" },
  { id: 2, src: stage02, label: "Zooming into the bacterial cell" },
  { id: 3, src: stage04, label: "Bacteria in focus" },
  { id: 4, src: stage07, label: "Antimicrobial peptide approaches" },
  { id: 5, src: stage08, label: "Peptide chain formation" },
  { id: 6, src: stage09, label: "Peptide attaches to bacteriophage" },
  { id: 7, src: stage10, label: "Bacteriophage activated" },
  { id: 8, src: stage11, label: "Infection begins" },
  { id: 9, src: stage12, label: "Bacterial breakdown" },
  { id: 10, src: stage13, label: "Mission accomplished" },
];

export function ParallaxHero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const stageRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (!containerRef.current || !viewportRef.current) return;

    const ctx = gsap.context(() => {
      // Create timeline pinned to viewport
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: containerRef.current,
          start: "top top",
          end: "bottom bottom",
          scrub: 1,
          pin: viewportRef.current,
          anticipatePin: 1,
        },
      });

      // Define zoom/pan for each stage (based on user's description)
      const stageAnimations = [
        { scale: 1, y: 0, opacity: 1 }, // Stage 1: Initial view
        { scale: 1.5, y: -10, opacity: 1 }, // Stage 2: Zoom into mask
        { scale: 2, y: -20, opacity: 1 }, // Stage 3: Bacteria shows up
        { scale: 1.5, y: -10, opacity: 1 }, // Stage 4: Zoom out to mask
        { scale: 1.8, y: -25, opacity: 1 }, // Stage 5: Peptide chain
        { scale: 2.2, y: -30, opacity: 1 }, // Stage 6: Peptide moves
        { scale: 2.5, y: -35, opacity: 1 }, // Stage 7: Stick on virus
        { scale: 2.2, y: -30, opacity: 1 }, // Stage 8: Virus shrinks
        { scale: 2.5, y: -40, opacity: 1 }, // Stage 9: Infects bacteria
        { scale: 3, y: -50, opacity: 1 }, // Stage 10: Bacteria breaks down
      ];

      // Initialize all stages
      stageRefs.current.forEach((stage, index) => {
        if (!stage) return;
        if (index === 0) {
          gsap.set(stage, { opacity: 1, scale: 1, y: 0 });
        } else {
          gsap.set(stage, { opacity: 0, scale: 0.9, y: 20 });
        }
      });

      // Create smooth transitions between stages
      stageRefs.current.forEach((stage, index) => {
        if (!stage || index === 0) return;

        const progress = index / stageRefs.current.length;
        const prevAnim = stageAnimations[index - 1];
        const currAnim = stageAnimations[index];

        // Fade out previous
        tl.to(
          stageRefs.current[index - 1],
          {
            opacity: 0,
            scale: prevAnim.scale * 1.05,
            y: prevAnim.y - 5,
            duration: 0.3,
            ease: "power2.inOut",
          },
          progress
        );

        // Fade in current with zoom
        tl.to(
          stage,
          {
            opacity: 1,
            scale: currAnim.scale,
            y: currAnim.y,
            duration: 0.5,
            ease: "power2.inOut",
          },
          progress + 0.1
        );

        // Hold the stage
        tl.to(
          stage,
          {
            scale: currAnim.scale,
            y: currAnim.y,
            duration: 0.4,
          },
          progress + 0.3
        );
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  // Reduce motion for accessibility
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mediaQuery.matches) {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative"
      style={{ height: "800vh" }}
    >
      {/* Fixed viewport */}
      <div
        ref={viewportRef}
        className="sticky top-0 h-screen w-full overflow-hidden bg-black"
      >
        {/* Stage layers - container with proper scaling origin */}
        <div className="relative h-full w-full flex items-center justify-center">
          {stages.map((stage, index) => (
            <div
              key={stage.id}
              ref={(el) => (stageRefs.current[index] = el)}
              className="absolute inset-0 flex items-center justify-center"
              style={{ 
                willChange: "transform, opacity",
                transformOrigin: "center center"
              }}
            >
              <img
                src={stage.src}
                alt={stage.label}
                className="w-auto h-full object-cover"
                style={{ 
                  maxWidth: "none",
                  transformOrigin: "center center"
                }}
              />
              <div className="absolute bottom-16 left-0 right-0 text-center z-10 pointer-events-none">
                <p className="text-base md:text-lg font-semibold text-white bg-black/50 backdrop-blur-sm py-3 px-6 rounded-full inline-block border border-white/20 shadow-lg">
                  {stage.label}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce z-20">
          <span className="text-sm text-white/80 font-medium">Scroll to explore</span>
          <svg
            className="w-6 h-6 text-white/80"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
