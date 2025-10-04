import { stringToSlug } from "../utils";

export function Footer() {
  const teamYear = import.meta.env.VITE_TEAM_YEAR || "2025";
  const teamName = import.meta.env.VITE_TEAM_NAME || "iGEM IIT Roorkee";
  const teamSlug = stringToSlug(teamName);

  // color tokens (kept inline so easy to tweak)
  const colors = {
    dark: "#0A3C7D",
    mid: "#1F66C1",
    light: "#4DA3FF",
    cardBg: "rgba(255,255,255,0.06)",
  };

  return (
    <footer
      aria-labelledby="footer-heading"
      style={{
        background: `linear-gradient(90deg, ${colors.dark}, ${colors.mid})`,
        color: "white",
        paddingTop: "3rem",
        paddingBottom: "2rem",
      }}
      className="footer"
    >
      <div style={{marginLeft: "5rem", marginRight: "5rem"}}>
        <div className="row gy-4 align-items-start">
          {/* LEFT: logo + address */}
          <div className="col-lg-4 col-md-6">
            <div
              style={{
                display: "flex",
                gap: "12px",
                alignItems: "center",
                marginBottom: "1rem",
              }}
            >
              <img
                src="https://static.igem.wiki/teams/6026/igem2025/igem-logo.webp"
                alt="iGEM IIT Roorkee logo"
                style={{ height: 48, width: "auto", borderRadius: 8 }}
              />
              <h3 id="footer-heading" style={{ margin: 0, fontSize: "1.25rem" }}>
                {teamName}
              </h3>
            </div>

            <address style={{ lineHeight: 1.5, opacity: 0.95 }}>
              The Department of Biosciences and Bioengineering,
              <br />
              Indian Institute of Technology Roorkee,
              <br />
              Roorkee, Uttarakhand
              <br />
              India - 247667
            </address>

            <div style={{ marginTop: "1.25rem" }}>
              <img
                src="https://static.igem.wiki/teams/6026/igem2025/main-building.webp"
                alt="IITR Main Building"
                style={{
                  maxWidth: "220px",
                  width: "100%",
                  height: "auto",
                  filter: "drop-shadow(0 6px 18px rgba(0,0,0,0.28))",
                  borderRadius: 8,
                }}
              />
            </div>
          </div>

          {/* MIDDLE: quick links broken into columns */}
          <div className="col-lg-5 col-md-6">
            <div className="row">
              {[
                {
                  heading: "Project",
                  items: ["Description", "Problem", "Solution", "Implementation", "Contribution"],
                },
                {
                  heading: "Human Practices",
                  items: ["Overview", "Education", "Integrated Human Practices"],
                },
                {
                  heading: "Wet Lab",
                  items: ["Overview", "Protocol", "Results", "Safety", "Proof Of Concept"],
                },
              ].map((chunk) => (
                <div key={chunk.heading} className="col-6 col-sm-4 mb-3">
                  <h5 style={{ fontWeight: 600 }}>{chunk.heading}</h5>
                  <ul style={{ fontSize: "0.9rem", listStyle: "none", padding: 0, margin: 0 }}>
                    {chunk.items.map((it) => (
                      <li key={it} style={{ margin: "0.45rem 0" }}>
                        <a
                          href={`${it.toLowerCase().replace(/\s+/g, "-")}`}
                          className="footer-link"
                          style={{ textDecoration: "none" }}
                        >
                          {it}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT: Contact + socials */}
          <div className="col-lg-3 col-md-12">
            <h5 style={{ fontWeight: 600 }}>Contact</h5>
            <p style={{ marginBottom: 4 }}>Department of Biosciences and Bioengineering</p>
            <p style={{ marginBottom: 4 }}>
              <strong>Email:</strong> <a className="footer-link" href="mailto:igem@iitr.ac.in">igem@iitr.ac.in</a>
            </p>
            <p style={{ marginBottom: 8 }}>
              <strong>Phone:</strong> +91 9438213300
            </p>

            <div style={{ marginTop: 12 }}>
              <div style={{ marginBottom: 6, fontWeight: 700 }}>Follow us</div>
              <div style={{ display: "flex", gap: 10 }}>
                {/* Inline SVG icons to avoid extra deps */}
                <a
                  href="https://www.instagram.com/igem_iitr/"
                  aria-label="Instagram"
                  className="icon-btn"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <rect x="2" y="2" width="20" height="20" rx="5" stroke="white" strokeWidth="1.2"/>
                    <circle cx="12" cy="12" r="3.2" stroke="white" strokeWidth="1.2"/>
                    <circle cx="17.5" cy="6.5" r="0.7" fill="white"/>
                  </svg>
                </a>

                <a
                  href="https://www.facebook.com/igemiitr/"
                  aria-label="Facebook"
                  className="icon-btn"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M15 8h2V5h-2c-1.66 0-3 1.34-3 3v1H10v3h2v7h3v-7h2.2l.3-3H15V8z" stroke="white" strokeWidth="1.2" fill="none"/>
                  </svg>
                </a>

                <a
                  href="https://x.com/igem_iitr"
                  aria-label="Twitter"
                  className="icon-btn"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M22 5.8c-.7.3-1.4.6-2.2.7a3.6 3.6 0 0 0-6.2 3.3A10.2 10.2 0 0 1 3.2 4.5a3.6 3.6 0 0 0 1.1 4.8c-.6 0-1.1-.2-1.6-.4v.1c0 1.6 1.2 3 2.7 3.3-.5.1-1 .1-1.6.05.45 1.4 1.8 2.4 3.4 2.45A7.3 7.3 0 0 1 2 19.5 10.3 10.3 0 0 0 8.8 21c6.2 0 9.6-5.2 9.6-9.8v-.45c.7-.4 1.3-1 1.9-1.65-.65.25-1.34.42-2.05.5z" stroke="white" strokeWidth="0.9" fill="none"/>
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>

        <hr style={{ borderColor: "rgba(255,255,255,0.08)", marginTop: "1.75rem" }} />

        {/* LICENSE & REPO */}
        <div className="row">
          <div className="col-md-9">
            <p style={{ margin: 0, fontSize: "0.9rem", opacity: 0.95 }}>
              <small>
                © {teamYear} - Content on this site is licensed under a{" "}
                <a
                  className="subfoot"
                  href="https://creativecommons.org/licenses/by/4.0/"
                  rel="license"
                >
                  Creative Commons Attribution 4.0 International license
                </a>
                .
              </small>
            </p>
            <p style={{ margin: 0, marginTop: 6 }}>
              <small>
                The repository used to create this website is available at{" "}
                <a href={`https://gitlab.igem.org/${teamYear}/${teamSlug}`} style={{ color: "white", textDecoration: "underline" }}>
                  gitlab.igem.org/{teamYear}/{teamSlug}
                </a>
                .
              </small>
            </p>
          </div>

          <div className="col-md-3 text-md-end" style={{ marginTop: 6 }}>
            <small style={{ opacity: 0.85 }}>Made with ❤️ • iGEM IIT Roorkee</small>
          </div>
        </div>
      </div>

      {/* Styles scoped to the footer */}
      <style>{`
        .footer a { color: white; }
        .footer .subfoot { color: rgba(255,255,255,0.95); text-decoration: underline; }
        .footer .footer-link {
          color: rgba(255,255,255,0.92);
          opacity: 0.95;
          transition: transform 0.22s ease, color 0.22s ease;
          display: inline-block;
        }
        .footer .footer-link:hover {
          color: ${colors.light};
          transform: translateY(-4px);
        }

        .icon-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          border-radius: 8px;
          background: rgba(255,255,255,0.04);
          transition: transform 0.18s ease, background 0.18s ease;
          text-decoration: none;
        }
        .icon-btn:hover {
          transform: translateY(-4px) scale(1.03);
          background: rgba(77,163,255,0.12);
        }

        /* Responsive behavior: collapse lists into stacked sections on small screens */
        @media (max-width: 767.98px) {
          .footer h5 { margin-top: 0.3rem; }
          .footer .col-6 { flex: 0 0 50%; max-width: 50%; }
        }

        /* subtle entrance animation */
        .footer {
          animation: footerFadeUp 0.6s cubic-bezier(.2,.8,.2,1);
        }
        @keyframes footerFadeUp {
          from { transform: translateY(8px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `}</style>
    </footer>
  );
}
