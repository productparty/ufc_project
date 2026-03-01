const path = require('path');
const ROOT = path.join(__dirname, '..');
const d = JSON.parse(require('fs').readFileSync(path.join(ROOT, 'results', 'db-dump.json'), 'utf8'));

// Check data completeness per event
const evMap = {};
d.events.forEach(e => { evMap[e.id] = e.name; });

const stats = {};
d.fights.forEach(f => {
    const name = evMap[f.eventId] || f.eventId;
    if (!stats[name]) stats[name] = {total: 0, withTap: 0, withDR: 0, withFM: 0, withUFC: 0, withFMExpanded: 0};
    stats[name].total++;
    const a = f.fighterA;
    const b = f.fighterB;
    if (a.tapology && a.tapology.consensus !== null) stats[name].withTap++;
    if (a.dratings && a.dratings.winPct !== null) stats[name].withDR++;
    if (a.fightMatrix && a.fightMatrix.cirrs !== null) stats[name].withFM++;
    if (a.ufcStats && a.ufcStats.slpm !== null) stats[name].withUFC++;
    if (a.fightmatrix && a.fightmatrix.eloK170) stats[name].withFMExpanded++;
});

Object.entries(stats).forEach(([name, s]) => {
    console.log(name);
    console.log(`  Fights: ${s.total} | Tapology: ${s.withTap} | DRatings: ${s.withDR} | FM CIRRS: ${s.withFM} | FM Expanded: ${s.withFMExpanded} | UFCStats: ${s.withUFC}`);
});

// Check results availability
const resultsByEvent = {};
d.results.forEach(r => {
    const name = evMap[r.eventId] || 'unknown';
    if (!resultsByEvent[name]) resultsByEvent[name] = 0;
    resultsByEvent[name]++;
});
console.log('\nResults per event:');
Object.entries(resultsByEvent).forEach(([name, count]) => console.log(`  ${name}: ${count}`));
