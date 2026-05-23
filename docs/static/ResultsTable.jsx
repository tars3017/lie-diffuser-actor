// ResultsTable.jsx — CALVIN results table.
function ResultsTable() {
  const rows = [
    { method: '3D Diffuser Actor (600K)',           vals: [92.2, 78.7, 63.9, 51.2, 41.2], avg: 3.270, ours: false, baseline: true },
    { method: 'LDA (600K) w/o GAT Encoder',         vals: [89.6, 78.0, 66.6, 55.7, 46.9], avg: 3.368, ours: false },
    { method: 'LDA (300K) w/o Lie Diffusion',       vals: [90.2, 80.3, 69.6, 58.5, 48.8], avg: 3.474, ours: false },
    { method: 'Lie Diffuser Actor (300K)',          vals: [93.7, 83.4, 70.3, 57.6, 46.2], avg: 3.512, ours: true },
  ];
  return (
    <div className="lda-table-wrap">
      <table className="lda-table">
        <thead>
          <tr>
            <th>Method</th>
            <th>SR1</th><th>SR2</th><th>SR3</th><th>SR4</th><th>SR5</th>
            <th>Avg. Length</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r.ours ? 'ours' : (r.baseline ? 'baseline' : '')}>
              <td className="method">
                {r.method}
                {r.baseline && <span className="lda-tag">baseline</span>}
              </td>
              {r.vals.map((v,j) => <td key={j} className="num">{v.toFixed(1)}</td>)}
              <td className="num avg">{r.avg.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="lda-table-cap">
        <span className="lda-fignum">Table 1.</span>
        <span className="lda-figcaption">Zero-shot CALVIN ABC→D. Lie Diffuser Actor improves average task length from 3.27 to 3.51, with consistent gains across all five sub-tasks.</span>
      </div>
    </div>
  );
}
