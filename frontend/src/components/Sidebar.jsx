import {
  Activity,
  BarChart3,
  Brain,
  ClipboardList,
  FileText,
  House,
  Info,
  MessageSquareText,
} from "lucide-react";

const navItems = [
  { key: "home",     label: "Dashboard",                icon: House,             badge: null  },
  { key: "brain",    label: "Brain Imaging Analysis",   icon: Brain,             badge: "AI"  },
  { key: "speech",   label: "Speech & Language",        icon: MessageSquareText, badge: "AI"  },
  { key: "combined", label: "Integrated Summary",       icon: BarChart3,         badge: null  },
  { key: "reports",  label: "Patient Reports",          icon: FileText,          badge: null  },
  { key: "about",    label: "About Platform",           icon: Info,              badge: null  },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      {/* Brand block */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <Activity size={20} />
        </div>
        <div>
          <div className="sidebar-brand-name">NeuroMind</div>
          <div className="sidebar-brand-tag">Clinical Platform</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav" aria-label="Clinical navigation">
        <div className="sidebar-nav-section">Clinical Tools</div>

        {navItems.map(({ key, label, icon: Icon, badge }) => {
          const isActive = activePage === key;
          return (
            <button
              key={key}
              type="button"
              className={`nav-item${isActive ? " active" : ""}`}
              onClick={() => onNavigate(key)}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon size={16} aria-hidden="true" />
              <span>{label}</span>
              {badge && (
                <span className="nav-badge" aria-label={`${badge} enabled`}>
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer note */}
      <div className="sidebar-footer">
        <div className="sidebar-info-note">
          <ClipboardList size={13} aria-hidden="true" />
          <p>
            Brain imaging and language assessments are conducted independently for clinical safety.
          </p>
        </div>
      </div>
    </aside>
  );
}
