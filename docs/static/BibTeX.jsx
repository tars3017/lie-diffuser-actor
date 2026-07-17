// BibTeX.jsx — citation block with copy-to-clipboard.
const BIBTEX_ENTRY = `@inproceedings{chuang2026lie,
  title     = {The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space},
  author    = {Chuang, Bing-Cheng and Chu, I-Hsuan and Lin, Bor-Jiun and Yang, YuanFu and Sun, Min and Lee, Chun-Yi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  publisher = {PMLR},
  address   = {Seoul, South Korea},
  year      = {2026}
}`;

function BibTeX() {
  const [copied, setCopied] = React.useState(false);

  React.useEffect(() => { lucide.createIcons(); }, [copied]);

  const copy = () => {
    navigator.clipboard.writeText(BIBTEX_ENTRY).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="lda-bibtex">
      <button className="lda-copy-btn" onClick={copy} aria-label="Copy BibTeX to clipboard">
        <i data-lucide={copied ? 'check' : 'copy'} width="13" height="13"></i>
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre>{BIBTEX_ENTRY}</pre>
    </div>
  );
}
