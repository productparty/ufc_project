const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'ufc-predictor-2026-02-08T04-07-22-367Z.json'), 'utf8'));

function normalizeMethod(m) {
    if (!m) return '';
    const u = m.toUpperCase().trim();
    if (['KO','TKO','KO/TKO'].includes(u)) return 'KO';
    if (['SUB','SUBMISSION'].includes(u)) return 'SUB';
    if (['DEC','DECISION','UD','SD','MD'].includes(u)) return 'DEC';
    return u;
}

function normalizeRound(r) {
    if (!r) return '';
    const u = String(r).toUpperCase().trim();
    if (u === 'DEC' || u === 'DECISION') return 'DEC';
    if (u.startsWith('R')) return u;
    if (!isNaN(parseInt(u))) return 'R' + u;
    return u;
}

const fights = data.allFights;
const results = data.allResults;
const fightMap = {};
fights.forEach(f => {
    fightMap[f.event + '|' + f.fighterA] = f;
    fightMap[f.event + '|' + f.fighterB] = f;
});

const wcOrder = ['HW', 'LHW', 'MW', 'WW', 'LW', 'FW', 'BW', 'FLW', 'WSW'];
const wcRounds = {};

results.forEach(r => {
    const fight = fightMap[r.event + '|' + r.winner];
    if (!fight) return;
    const wc = fight.weightClass || 'UNK';
    const method = normalizeMethod(r.method);
    if (method === 'DEC') return; // Only finishes

    const round = normalizeRound(r.round);
    if (!wcRounds[wc]) wcRounds[wc] = { R1: 0, R2: 0, R3: 0, R4: 0, R5: 0, total: 0, fights: [] };
    wcRounds[wc].total++;
    if (wcRounds[wc][round] !== undefined) wcRounds[wc][round]++;
    wcRounds[wc].fights.push(`${round} ${method} ${r.winner.split(' ').pop()}`);
});

console.log('='.repeat(80));
console.log('FINISH ROUND BY WEIGHT CLASS (our 17 finishes)');
console.log('='.repeat(80));
console.log();
console.log(`${'WC'.padEnd(6)} ${'Total'.padEnd(7)} ${'R1'.padEnd(10)} ${'R2'.padEnd(10)} ${'R3'.padEnd(10)} ${'Avg Rd'.padEnd(8)} Fights`);
console.log('─'.repeat(80));

let allR1 = 0, allR2 = 0, allR3 = 0, allTotal = 0;

wcOrder.forEach(wc => {
    const d = wcRounds[wc];
    if (!d || d.total === 0) return;

    // Calculate average round
    const avgRd = (d.R1 * 1 + d.R2 * 2 + d.R3 * 3 + d.R4 * 4 + d.R5 * 5) / d.total;

    allR1 += d.R1; allR2 += d.R2; allR3 += d.R3; allTotal += d.total;

    const r1Pct = d.R1 > 0 ? `${d.R1} (${(d.R1/d.total*100).toFixed(0)}%)` : '0';
    const r2Pct = d.R2 > 0 ? `${d.R2} (${(d.R2/d.total*100).toFixed(0)}%)` : '0';
    const r3Pct = d.R3 > 0 ? `${d.R3} (${(d.R3/d.total*100).toFixed(0)}%)` : '0';

    console.log(`${wc.padEnd(6)} ${String(d.total).padEnd(7)} ${r1Pct.padEnd(10)} ${r2Pct.padEnd(10)} ${r3Pct.padEnd(10)} ${avgRd.toFixed(1).padEnd(8)} ${d.fights.join(', ')}`);
});

const allAvg = (allR1 * 1 + allR2 * 2 + allR3 * 3) / allTotal;
console.log('─'.repeat(80));
console.log(`${'ALL'.padEnd(6)} ${String(allTotal).padEnd(7)} ${(allR1 + ' (' + (allR1/allTotal*100).toFixed(0) + '%)').padEnd(10)} ${(allR2 + ' (' + (allR2/allTotal*100).toFixed(0) + '%)').padEnd(10)} ${(allR3 + ' (' + (allR3/allTotal*100).toFixed(0) + '%)').padEnd(10)} ${allAvg.toFixed(1)}`);

console.log('\n\n--- Current Model Round Thresholds (3-round fights) ---');
console.log('Higher threshold = harder to predict that round = later round bias\n');

const thresholds3 = {
    'HW':   { R1: 42.0, R2: 33.0 },
    'LHW':  { R1: 45.0, R2: 36.0 },
    'MW':   { R1: 48.0, R2: 39.0 },
    'WW':   { R1: 50.0, R2: 41.0 },
    'LW':   { R1: 52.0, R2: 43.0 },
    'FW':   { R1: 52.0, R2: 43.0 },
    'BW':   { R1: 54.0, R2: 45.0 },
    'FLW':  { R1: 55.0, R2: 46.0 },
};

console.log(`${'WC'.padEnd(6)} ${'R1 thresh'.padEnd(12)} ${'R2 thresh'.padEnd(12)} ${'Actual R1%'.padEnd(12)} ${'Actual R2%'.padEnd(12)} ${'Suggestion'}`);
wcOrder.forEach(wc => {
    const t = thresholds3[wc];
    const d = wcRounds[wc];
    if (!t) return;
    const r1Pct = d ? (d.R1/d.total*100).toFixed(0) + '%' : 'n/a';
    const r2Pct = d ? (d.R2/d.total*100).toFixed(0) + '%' : 'n/a';

    let suggestion = '';
    if (d) {
        if (d.R2 > d.R1) suggestion = '← R2 dominant, raise R1 threshold';
        else if (d.R1 > d.R2) suggestion = '← R1 dominant, keep or lower R1 threshold';
        else suggestion = '← even';
    }

    console.log(`${wc.padEnd(6)} ${String(t.R1).padEnd(12)} ${String(t.R2).padEnd(12)} ${r1Pct.padEnd(12)} ${r2Pct.padEnd(12)} ${suggestion}`);
});
