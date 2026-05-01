// RobotGrid.jsx — 2×2 grid of the four real-robot tasks.
// Videos play at 2× speed (a "2×" badge marks the upper-right of each tile).
function RobotGrid() {
  const tasks = [
    { id: 'doll',  name: 'Move Doll Platform', desc: 'Stable 6-DOF transport.',          ours: 100, base: 90, video: 'static/videos/put_doll.mp4' },
    { id: 'block', name: 'Put Block in Box',   desc: 'Tight-tolerance insertion.',       ours:  75, base: 80, video: 'static/videos/put_block.mp4' },
    { id: 'sort',  name: 'Sort Blocks',        desc: 'Multi-step orientation control.',  ours:  75, base: 55, video: 'static/videos/sort_blocks.mp4' },
    { id: 'stack', name: 'Stack Cups',         desc: 'Sub-centimeter alignment.',        ours:  60, base: 55, video: 'static/videos/stack_cups.mp4' },
  ];

  const setRate2x = (e) => {
    if (e && e.currentTarget) e.currentTarget.playbackRate = 2.0;
  };

  return (
    <div className="lda-robot-grid">
      {tasks.map(t => (
        <div className="lda-robot-tile" key={t.id}>
          <div className="lda-robot-media">
            {t.video ? (
              <>
                <video
                  className="lda-robot-video"
                  autoPlay muted loop playsInline controls preload="metadata"
                  onLoadedMetadata={setRate2x}
                  onRateChange={setRate2x}
                >
                  <source src={t.video} type="video/mp4" />
                </video>
                <div className="lda-robot-speed-tag" aria-label="Playback at 2× speed">2×</div>
              </>
            ) : (
              <div className="lda-placeholder-grid"></div>
            )}
            <div className="lda-robot-tag">{t.video ? t.video.split('/').pop() : `task ${t.id}.mp4`}</div>
          </div>
          <div className="lda-robot-info">
            <div className="lda-robot-name">{t.name}</div>
            <div className="lda-robot-desc">{t.desc}</div>
            <div className="lda-robot-stats">
              <div className="lda-stat ours">
                <span className="lda-stat-label">LDA</span>
                <span className="lda-stat-val">{t.ours}%</span>
              </div>
              <div className="lda-stat base">
                <span className="lda-stat-label">Baseline</span>
                <span className="lda-stat-val">{t.base}%</span>
              </div>
              <div className={`lda-stat-delta ${t.ours >= t.base ? 'pos' : 'neg'}`}>
                {t.ours >= t.base ? '+' : '−'}{Math.abs(t.ours - t.base)}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
