// Nav.jsx — sticky top nav. Italic serif mark + Inter section links.
function Nav() {
  const [hash, setHash] = React.useState(typeof window !== 'undefined' ? window.location.hash || '#abstract' : '#abstract');
  const items = [
    { id: 'abstract', label: 'Abstract' },
    { id: 'method', label: 'Method' },
    { id: 'results', label: 'Results' },
    { id: 'realrobot', label: 'Real Robot' },
    { id: 'bibtex', label: 'BibTeX' },
  ];
  return (
    <nav className="lda-nav">
      <a href="#top" className="lda-mark" onClick={() => setHash('#top')}>
        Lie Diffuser Actor
      </a>
      <div className="lda-nav-links">
        {items.map(it => (
          <a key={it.id} href={`#${it.id}`} className={hash === `#${it.id}` ? 'active' : ''} onClick={() => setHash(`#${it.id}`)}>
            {it.label}
          </a>
        ))}
      </div>
    </nav>
  );
}
