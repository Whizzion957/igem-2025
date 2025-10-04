import Container from "react-bootstrap/Container";
import Nav from "react-bootstrap/Nav";
import BootstrapNavbar from "react-bootstrap/Navbar";
import NavDropdown from "react-bootstrap/NavDropdown";
import { Link } from "react-router-dom";
import Pages from "../pages.ts";

export function Navbar() {
  const pages = Pages.map((item, pageIndex) => {
    if ("folder" in item && item.folder) {
      const folderItems = item.folder.map((subpage, subpageIndex) => {
        if (subpage.path) {
          return (
            <NavDropdown.Item
              key={`subpage-${pageIndex}-${subpageIndex}`}
              as={Link}
              to={subpage.path}
            >
              {subpage.name}
            </NavDropdown.Item>
          );
        }
      });
      return (
        <NavDropdown
          key={`page-${pageIndex}`}
          title={item.name}
          id={`page-${pageIndex}`}
        >
          {folderItems}
        </NavDropdown>
      );
    } else if ("path" in item && item.path) {
      return (
        <Nav.Link key={`page-${pageIndex}`} as={Link} to={item.path}>
          {item.name}
        </Nav.Link>
      );
    }
  });

  return (
    <>
      <BootstrapNavbar
        expand="lg"
        fixed="top"
        style={{
          background: "linear-gradient(90deg, #0A3C7D, #1F66C1)",
          backdropFilter: "blur(10px)",
          boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
        }}
      >
        <Container>
          <BootstrapNavbar.Brand
            style={{
              fontWeight: "bold",
              fontSize: "1.5rem",
              color: "white",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <img
              src="https://static.igem.wiki/teams/6026/igem2025/amplify-logo.webp" // <- replace with your actual logo path
              alt="Logo"
              style={{
                height: "55px",
                width: "auto",
                objectFit: "contain",
              }}
            />
          </BootstrapNavbar.Brand>

          <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
          <BootstrapNavbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">{pages}</Nav>
          </BootstrapNavbar.Collapse>
        </Container>
      </BootstrapNavbar>
      <style>
        {`
          .navbar-nav .nav-link {
            color: white !important;
            font-weight: 500;
            margin: 0 12px;
            transition: all 0.3s ease;
            position: relative;
          }

          .navbar-nav .nav-link:hover {
            color: #4DA3FF !important;
            transform: translateY(-3px);
          }

          .dropdown-menu {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 10px;
            animation: fadeIn 0.3s ease;
          }

          .dropdown-item {
            color: white !important;
            transition: 0.2s;
          }

          .dropdown-item:hover {
            background: rgba(77, 163, 255, 0.2);
            border-radius: 8px;
          }

          .nav-item.dropdown:hover .dropdown-menu {
            display: block;
            margin-top: 0;
          }

          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}
      </style>
    </>
  );
}
