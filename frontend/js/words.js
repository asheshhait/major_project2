export function currentWordCounter(current, total) {
  return `Word ${current} / ${total}`;
}

export function splitWordIntoChars(wordText) {
  const text = wordText?.trim() || "";
  if (!text) return [];

  if (typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter("bn", { granularity: "grapheme" });
    return Array.from(segmenter.segment(text), ({ segment }) => segment);
  }

  return splitBanglaGraphemes(text);
}

function splitBanglaGraphemes(text) {
  const clusters = [];
  const marks = /\p{Mark}/u;
  const codePoints = Array.from(text);

  for (let index = 0; index < codePoints.length; index += 1) {
    const codePoint = codePoints[index];
    if (/\s/u.test(codePoint)) continue;

    let cluster = codePoint;
    while (index + 1 < codePoints.length) {
      const next = codePoints[index + 1];
      if (marks.test(next)) {
        cluster += next;
        index += 1;
        if (next === "\u09CD" && index + 1 < codePoints.length) {
          cluster += codePoints[index + 1];
          index += 1;
        }
        continue;
      }
      break;
    }

    clusters.push(cluster);
  }

  return clusters;
}
