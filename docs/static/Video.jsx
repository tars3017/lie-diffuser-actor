// Video.jsx — teaser video frame with play overlay.
function Video({ src, poster, label, autoPlay }) {
  const [playing, setPlaying] = React.useState(false);
  const shouldPlay = Boolean(src && (autoPlay || playing));

  return (
    <div className="lda-video" id="video">
      {shouldPlay ? (
        <video src={src} poster={poster} controls autoPlay muted loop playsInline preload="metadata" className="lda-video-el" />
      ) : (
        <button className="lda-video-poster" onClick={() => setPlaying(true)}>
          <div className="lda-video-bg">
            {/* placeholder: stripe pattern + label */}
            <div className="lda-placeholder-stripes"></div>
            <div className="lda-video-label">{label || 'Teaser video · 2 min'}</div>
          </div>
          <div className="lda-video-play">
            <i data-lucide="play" style={{width:28,height:28,strokeWidth:1.5}}></i>
          </div>
        </button>
      )}
    </div>
  );
}
