// Figure.jsx — figure container with mono fignum + serif italic caption.
function Figure({ num, caption, children, wide, bare }) {
  if (bare) {
    return (
      <figure className={`lda-figure bare ${wide ? 'wide' : ''}`}>
        {children}
        <figcaption>
          <span className="lda-fignum">Figure {num}.</span>
          <span className="lda-figcaption">{caption}</span>
        </figcaption>
      </figure>
    );
  }

  return (
    <figure className={`lda-figure ${wide ? 'wide' : ''}`}>
      <div className="lda-figure-canvas">{children}</div>
      <figcaption>
        <span className="lda-fignum">Figure {num}.</span>
        <span className="lda-figcaption">{caption}</span>
      </figcaption>
    </figure>
  );
}

function FigurePlaceholder({ label, height = 280 }) {
  return (
    <div className="lda-fig-placeholder" style={{ height }}>
      <div className="lda-fig-placeholder-grid"></div>
      <div className="lda-fig-placeholder-label">{label}</div>
    </div>
  );
}

// Inline SVG schematic of Euclidean vs. Lie diffusion (Figure 1 idea).
function ManifoldSchematic() {
  return (
    <svg viewBox="0 0 800 280" className="lda-svg-fig" xmlns="http://www.w3.org/2000/svg">
      {/* TOP — Euclidean: linear interpolation cuts the sphere */}
      <g transform="translate(0,0)">
        <text x="20" y="28" className="lda-svg-label">Euclidean diffusion · ℝ¹²</text>
        <text x="20" y="46" className="lda-svg-sub">Linear noise leaves the manifold.</text>
        <ellipse cx="400" cy="80" rx="110" ry="32" fill="none" stroke="#C9C6BD" strokeWidth="1"/>
        <ellipse cx="400" cy="80" rx="110" ry="32" fill="none" stroke="#0F6E68" strokeWidth="0.6" strokeDasharray="2 3" opacity="0.5"/>
        {/* jagged red baseline path */}
        <path d="M310 70 L 340 100 L 370 50 L 410 110 L 450 55 L 490 100 L 510 70" fill="none" stroke="#B83A2A" strokeWidth="1.6" strokeLinejoin="round"/>
        <circle cx="310" cy="70" r="3.5" fill="#B83A2A"/>
        <circle cx="510" cy="70" r="3.5" fill="#B83A2A"/>
        <text x="295" y="62" className="lda-svg-tag">g₀</text>
        <text x="515" y="62" className="lda-svg-tag">gₜ</text>
      </g>
      {/* BOTTOM — Lie diffusion: trajectory hugs the manifold */}
      <g transform="translate(0,140)">
        <text x="20" y="28" className="lda-svg-label">Lie diffusion · SE(3)</text>
        <text x="20" y="46" className="lda-svg-sub">Tangent twists ω ∈ se(3) retracted via exp(·).</text>
        <ellipse cx="400" cy="80" rx="110" ry="32" fill="none" stroke="#C9C6BD" strokeWidth="1"/>
        {/* smooth teal arc on the surface */}
        <path d="M310 80 C 350 70, 450 70, 510 80" fill="none" stroke="#0F6E68" strokeWidth="1.8"/>
        <circle cx="310" cy="80" r="3.5" fill="#0F6E68"/>
        <circle cx="510" cy="80" r="3.5" fill="#0F6E68"/>
        {/* tangent twist annotation */}
        <line x1="380" y1="73" x2="408" y2="60" stroke="#14171A" strokeWidth="0.8" markerEnd="url(#arrhead)"/>
        <text x="412" y="58" className="lda-svg-tag">ω</text>
        <text x="295" y="72" className="lda-svg-tag">g₀</text>
        <text x="515" y="72" className="lda-svg-tag">gₜ</text>
      </g>
      <defs>
        <marker id="arrhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" fill="#14171A"/>
        </marker>
      </defs>
    </svg>
  );
}
