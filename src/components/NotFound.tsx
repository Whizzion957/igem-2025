import { Link } from "react-router-dom";

export function NotFound() {
  return (
    <main
      role="main"
      aria-labelledby="notfound-title"
      style={{
        background: "#ffffff",
        color: "#0A3C7D", // logo-consistent blue for accents
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <div
        className="nf-card"
        style={{
          width: "100%",
          maxWidth: 820,
          textAlign: "center",
          padding: "2.25rem",
          borderRadius: 12,
          // keep card subtle on white background
          border: "1px solid rgba(10,60,125,0.06)",
          boxShadow: "0 8px 30px rgba(10,60,125,0.04)",
        }}
      >
        <h1
          id="notfound-title"
          className="nf-404"
          style={{
            margin: 0,
            fontSize: "clamp(48px, 14vw, 120px)",
            lineHeight: 1,
            fontWeight: 800,
            color: "#0A3C7D",
          }}
        >
          404
        </h1>

        <h2
          style={{
            marginTop: "0.5rem",
            marginBottom: "0.5rem",
            fontSize: "1.25rem",
            fontWeight: 600,
          }}
        >
          Page not found
        </h2>

        <p style={{ margin: 0, color: "rgba(10,60,125,0.85)" }}>
          The page you’re looking for can’t be found. It may have been moved or the
          link is broken.
        </p>

        <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem", justifyContent: "center", flexWrap: "wrap" }}>
          <Link to="/" className="nf-btn primary" aria-label="Back to Home">
            Back to Home
          </Link>
        </div>

        <p style={{ marginTop: "1rem", color: "rgba(10,60,125,0.6)", fontSize: "0.9rem" }}>
          Try returning home or exploring other pages.
        </p>
        <div style={{ marginTop: 18, textAlign: "center", zIndex: 2 }}>
          <small style={{ color: "rgba(10,60,125,0.6)" }}>© {new Date().getFullYear()} • iGEM IIT Roorkee</small>
        </div>
      </div>

      <style>{`
        /* subtle pop animation for the large 404 */
        .nf-404 {
          display: inline-block;
          animation: nfPop 900ms cubic-bezier(.2,.9,.2,1);
          transform-origin: center;
        }
        @keyframes nfPop {
          0% { transform: scale(.92); opacity: 0; }
          40% { transform: scale(1.03); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }

        /* buttons */
        .nf-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 0.6rem 1rem;
          border-radius: 8px;
          font-weight: 600;
          text-decoration: none;
          transition: transform 150ms ease, box-shadow 150ms ease, background 150ms ease;
        }

        .nf-btn.primary {
          background: linear-gradient(90deg, #4DA3FF, #1F66C1);
          color: white;
          box-shadow: 0 8px 18px rgba(31,102,193,0.12);
        }
        .nf-btn.primary:hover {
          transform: translateY(-4px);
          box-shadow: 0 14px 30px rgba(31,102,193,0.16);
        }

        .nf-btn.ghost {
          background: transparent;
          color: #0A3C7D;
          border: 1px solid rgba(10,60,125,0.12);
        }
        .nf-btn.ghost:hover {
          transform: translateY(-3px);
          background: rgba(31,102,193,0.04);
        }

        /* small screens: tighten spacing */
        @media (max-width: 520px) {
          .nf-card { padding: 1.25rem; }
          .nf-btn { width: 100%; justify-content: center; }
        }
      `}</style>
    </main>
  );
}
