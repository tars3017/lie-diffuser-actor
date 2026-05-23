// Hero.jsx — paper title, authors, affiliations, action buttons.
function ActionButton({ icon, label, primary, onClick, href, disabled, title }) {
  const cls = `lda-btn ${primary ? 'lda-btn-primary' : 'lda-btn-secondary'}${disabled ? ' is-disabled' : ''}`;
  const inner = (<><span>{label}</span><i data-lucide={icon} style={{width:16,height:16,strokeWidth:1.75}}></i></>);
  if (disabled) return <button className={cls} disabled title={title}>{inner}</button>;
  return href ? <a href={href} className={cls} onClick={onClick} title={title}>{inner}</a>
              : <button className={cls} onClick={onClick} title={title}>{inner}</button>;
}

function Hero() {
  // marks: "1" ntu · "2" nycu · "3" nthu · "*" equal contribution
  const authors = [
    { name: "Bing-Cheng Chuang", marks: "1,*" },
    { name: "I-Hsuan Chu",       marks: "1,*" },
    { name: "Bor-Jiun Lin",      marks: "1"   },
    { name: "YuanFu Yang",       marks: "2"   },
    { name: "Min Sun",           marks: "3"   },
    { name: "Chun-Yi Lee",       marks: "1"   },
  ];
  return (
    <header id="top" className="lda-hero">
      <div className="lda-eyebrow lda-venue">
        <img src="https://icml.cc/static/core/img/icml-navbar-logo.svg" alt="ICML" />
        <span>2026</span>
      </div>

      <h1 className="lda-title">
        The Lie We Tell:<br/>
        <span className="lda-title-sub">Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space</span>
      </h1>

      <div className="lda-authors">
        {authors.map(({ name, marks }) => (
          <div className="lda-author" key={name}>
            <div className="lda-author-name">
              {name}<sup>{marks}</sup>
            </div>
          </div>
        ))}
      </div>
      <div className="lda-affil">
        <div><sup>1</sup>Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan</div>
        <div><sup>2</sup>Institute of Artificial Intelligence Innovation, National Yang Ming Chiao Tung University, Hsinchu, Taiwan</div>
        <div><sup>3</sup>Department of Electrical Engineering, National Tsing Hua University, Hsinchu, Taiwan</div>
        <div style={{marginTop: 'var(--s-2)'}}><sup>*</sup>Equal contribution.</div>
      </div>

      <div className="lda-lab">
        <img src="https://www.ntu.edu.tw/images/about/emblem_1.png" alt="National Taiwan University" />
        <img src="https://elsalab.ai/imgs/icons/elsa-lab.png" alt="ELSA Lab" />
      </div>

      <div className="lda-actions">
        <ActionButton primary icon="file-text" label="Paper (arXiv soon)" disabled title="arXiv link will be added after the camera-ready release" />
        <ActionButton icon="code" label="Code" href="https://github.com/tars3017/lie-diffuser-actor" />
        <ActionButton icon="quote" label="BibTeX" href="#bibtex" />
      </div>

      <div className="lda-tldr">
        <span className="lda-tldr-label">TL;DR</span>
        <p>
          Diffusion-based VLA policies treat <span className="glyph">SE(3)</span> poses as flat <span className="glyph">ℝ¹²</span> vectors — a geometric error we call the <em>Euclidean Fallacy</em>. <strong>Lie Diffuser Actor</strong> diffuses intrinsically on the manifold via the exponential map, eliminating manifold drift and improving CALVIN ABC→D from <span className="glyph">3.27 → 3.51</span> (+7.3%).
        </p>
      </div>
    </header>
  );
}
