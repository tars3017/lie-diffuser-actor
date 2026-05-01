// BibTeX.jsx — code block with copy-to-clipboard.
function BibTeX() {
  const cite = `@inproceedings{lda2026,
  title     = {The Lie We Tell: Correcting the Euclidean Fallacy
               in Vision-Language-Action Policies via
               Score Matching on Tangent Space},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference
               on Machine Learning (ICML)},
  year      = {2026}
}`;
  const [copied, setCopied] = React.useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(cite);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <div className="lda-bibtex">
      <button className="lda-copy-btn" onClick={copy}>
        <i data-lucide={copied ? 'check' : 'copy'} style={{width:14,height:14,strokeWidth:1.75}}></i>
        <span>{copied ? 'Copied' : 'Copy'}</span>
      </button>
      <pre>{cite}</pre>
    </div>
  );
}
