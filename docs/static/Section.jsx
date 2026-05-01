// Section.jsx — generic section frame: eyebrow + heading + content.
function Section({ id, eyebrow, title, narrow, children }) {
  return (
    <section id={id} className={`lda-section ${narrow ? 'narrow' : ''}`}>
      {eyebrow && <div className="lda-eyebrow">{eyebrow}</div>}
      {title && <h2 className="lda-h2">{title}</h2>}
      <div className="lda-section-body">{children}</div>
    </section>
  );
}
