import { Link } from 'react-router-dom';
import { ShieldCheck, Github, Activity, Home, BookOpen } from 'lucide-react';
import './Navbar.css';

export function NavBar() {
    return (
        <nav className="navbar">
            <div className="nav-content">
                
                <div className="nav-left">
                    <div className="nav-logo">
                        <ShieldCheck size={28} className="logo-icon" />
                        <span className="logo-text">Domain<span className="highlight">Scanner</span></span>
                    </div>


                    <div className="status-indicator">
                        <Activity size={16} />
                        <span>Online</span>
                        <div className="dot"></div>
                    </div>
                </div>

                
                <div className="nav-right">
                    <ul className="nav-links">
                        <li>
                            <Link to="/"><Home size={18} /> Home</Link>
                        </li>
                        <li>
                            <Link to="/docs"><BookOpen size={18} /> Docs</Link>
                        </li>
                    </ul>

                    
                    <a href="https://github.com/LHLobato/Domain-Attack-Detection-API/" target="_blank" className="github-btn">
                        <Github size={20} />
                        <span>GitHub</span>
                    </a>
                </div>

            </div>
        </nav>
    );
}