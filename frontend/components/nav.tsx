import Link from "next/link";

export function Nav() {
  return (
    <nav className="top-nav" aria-label="Primary">
      <div className="brand-mark">QuakeCore</div>
      <div className="nav-links">
        <Link href="/">Chat</Link>
        <Link href="/settings">Settings</Link>
        <Link href="/skills">Skills</Link>
      </div>
    </nav>
  );
}

