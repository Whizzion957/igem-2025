
export function Members() {
  const teamSections = [
    {
      title: "Team Leader",
      members: [
        {
          id: 1,
          name: "Adit Khare",
          role: "Team Leader",
          img: "https://static.igem.wiki/teams/6026/igem2025/teammembers/adit-khare-old.webp",
          intro:
            "Anastasia is passionate about molecular biotechnology and loves agarose gels. She helps design experiments, curate the wiki, and keeps the team organised.",
          field: "Department of Biosciences and Bioengineering",
        },
      ],
    },
    {
      title: "Additional Leaders",
      members: [
        {
          id: 2,
          name: "Hima Shashidhar",
          role: "Co-Leader",
          img: "https://static.igem.wiki/teams/6026/igem2025/teammembers/hima-old.webp",
          intro:
            "Hey all! I an Hima, a 4th year at IITR. iGEM has been an important part of my jouney to loving all things syn bio :)",
          field: "Department of Biosciences and Bioengineering",
        },
      ],
    },
    {
      title: "Head of Verticals",
      members: [
        {
          id: 3,
          name: "Prisha Sinha",
          role: "Drylab Head",
          img: "https://static.igem.wiki/teams/6026/igem2025/teammembers/prisha-old.webp",
          intro:
            "Hi I am Prisha ! I am passionate about the world of synthetic biology and its impact in the real world.",
          field: "Department of Biosciences and Bioengineering",
        },
      ],
    },
    {
      title: "Members",
      members: [],
    },
    {
      title: "Advisors",
      members: [],
    },
    {
      title: "Principal Instructors",
      members: [],
    },
  ];

  return (
    <>
      <style>{`
        .team-photo-container {
          width: 100%;
          margin-bottom: 40px;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        }
        
        .team-photo-container img {
          width: 100%;
          height: auto;
          display: block;
        }
        
        .team-section {
          margin-bottom: 60px;
          text-align: center;
        }
        
        .team-section-title {
          font-size: 2rem;
          font-weight: 700;
          margin-bottom: 24px;
          padding-bottom: 12px;
          border-bottom: 3px solid #0ea5e9;
          color: #1e293b;
          display: inline-block;
          
        }
        
        .flip-card-container {
          perspective: 1000px;
          width: 100%;
          max-width: 500px;
          height: 380px;
        }
        
        .flip-card {
          position: relative;
          width: 100%;
          height: 100%;
          transition: transform 0.6s;
          transform-style: preserve-3d;
        }
        
        .flip-card-container:hover .flip-card {
          transform: rotateY(180deg);
        }
        
        .flip-card-front,
        .flip-card-back {
          position: absolute;
          width: 100%;
          height: 100%;
          backface-visibility: hidden;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        }
        
        .flip-card-front {
          background: linear-gradient(135deg, #0A3C7D 0%, #4DA3FF 100%);
          display: flex;
        }
        
        .flip-card-back {
          background-color: #0A3C7D;
          transform: rotateY(180deg);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 30px;
          color: white;
        }
        
        .member-photo-container {
          padding: 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }
        
        .member-photo {
          width: 150px;
          height: 220px;
          border-radius: 10px;
          overflow: hidden;
          border: 4px solid rgba(255, 255, 255, 0.3);
          background: white;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .member-photo img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }
        
        .member-info {
          flex: 1;
          padding: 20px 20px 20px 0;
          color: white;
          display: flex;
          flex-direction: column;
          justify-content: center;
          position: relative;
        }
        
        .member-role {
          font-size: 1.1rem;
          font-weight: 700;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .member-details {
          margin-top: 8px;
        }
        
        .member-detail-item {
          margin-bottom: 12px;
        }
        
        .member-detail-label {
          font-weight: 600;
          font-size: 0.875rem;
          margin-bottom: 4px;
          opacity: 0.95;
        }
        
        .member-detail-value {
          font-size: 0.875rem;
          opacity: 0.85;
          line-height: 1.4;
        }
        
        .member-name {
          position: absolute;
          bottom: 12px;
          left: 20px;
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(8px);
          color: white;
          font-weight: 500;
          font-size: 1.1rem;
          padding: 1px 15px;
          border-radius: 10px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
          white-space: nowrap;
        }
        
        .member-intro {
          text-align: center;
          line-height: 1.6;
          font-size: 1rem;
        }
        
        .member-intro-name {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 16px;
        }
        
        .cards-grid {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          gap: 32px;
          padding: 20px 0;
        }
        
        @media (max-width: 768px) {
          .cards-grid {
            gap: 16px;
            height: 550px;
          }
          .flip-card {
            height: 550px;
          }
          .member-photo-container {
            height: 2000px;
          }
          .team-card {
            width: 100%;
            max-width: 350px;
          }
          
          .flip-card-container {
            height: 280px;
            max-width: 100%;
          }
          
          .flip-card-front {
            flex-direction: column;
          }
          
          .member-photo-container {
            padding: 16px;
          }
          
          .member-photo {
            display: flex;
            flex-direction: column;
            align-items: center;
          }
          
          .member-info {
            padding: 0 20px 20px 20px;
          }
          .member-name {
            position: static;            /* no longer absolute */
            transform: none;
            margin-top: 10px;            /* space below photo */
            color: #0A3C7D;
            font-size: 1rem;
            display: inline-block;
            padding: 6px 14px;
            border-radius: 6px;
            border: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
          }
          
        }
      `}</style>

      <div className="team-photo-container">
        <img src="https://static.igem.wiki/teams/6026/igem2025/igem-team-2024.webp" alt="Team Photo" />
      </div>

      {teamSections.map((section) => (
        <div key={section.title} className="team-section">
          <h2 className="team-section-title" >{section.title}</h2>
          
          {section.members.length > 0 ? (
            <div className="cards-grid">
              {section.members.map((member) => (
                <div key={member.id} className="flip-card-container">
                  <div className="flip-card">
                    {/* Front of card */}
                    <div className="flip-card-front">
                      <div className="member-photo-container">
                        <div className="member-photo">
                          <img src={member.img} alt={member.name} />
                          <div className="member-name">{member.name}</div>
                        </div>
                      </div>

                      <div className="member-info">
                        <div className="member-role">{member.role}</div>

                        <div className="member-details">
                          <div className="member-detail-item">
                            <div className="member-detail-label">Field of Study:</div>
                            <div className="member-detail-value">{member.field}</div>
                          </div>

                          <div className="member-detail-item">
                            <div className="member-detail-label"></div>
                            <div className="member-detail-value"></div>
                          </div>

                          <div className="member-detail-item">
                            <div className="member-detail-label"></div>
                            <div className="member-detail-value"></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Back of card */}
                    <div className="flip-card-back">
                      <div className="member-intro">
                        <div className="member-intro-name">{member.name}</div>
                        <p>{member.intro}</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: '#64748b', fontStyle: 'italic' }}>No members in this section yet.</p>
          )}
        </div>
      ))}
    </>
  );
}