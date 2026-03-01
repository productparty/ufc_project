/**
 * Weight class vs method/round correlation analysis
 */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'ufc-predictor-2026-02-08T04-07-22-367Z.json'), 'utf8'));

function normalizeMethod(method) {
    if (!method) return '';
    const m = method.toUpperCase().trim();
    if (['KO', 'TKO', 'KO/TKO'].includes(m)) return 'KO';
    if (['SUB', 'SUBMISSION'].includes(m)) return 'SUB';
    if (['DEC', 'DECISION', 'UD', 'SD', 'MD'].includes(m)) return 'DEC';
    return m;
}

function normalizeRound(round) {
    if (!round) return '';
    const r = String(round).toUpperCase().trim();
    if (r === 'DEC' || r === 'DECISION') return 'DEC';
    if (r.startsWith('R')) return r;
    if (!isNaN(parseInt(r))) return 'R' + r;
    return r;
}

const fights = data.allFights;
const results = data.allResults;
const predictions = data.allPredictions;

// Build fight lookup
const fightMap = {};
fights.forEach(f => {
    fightMap[f.event + '|' + f.fighterA] = f;
    fightMap[f.event + '|' + f.fighterB] = f;
});

// Analyze results by weight class
const wcData = {};
const overall = { total: 0, ko: 0, sub: 0, dec: 0, r1: 0, r2: 0, r3: 0, r4: 0, r5: 0 };

results.forEach(r => {
    const fight = fightMap[r.event + '|' + r.winner];
    if (!fight) return;

    const wc = fight.weightClass || 'UNK';
    if (!wcData[wc]) wcData[wc] = { total: 0, ko: 0, sub: 0, dec: 0, r1: 0, r2: 0, r3: 0, r4: 0, r5: 0, fights: [] };

    const method = normalizeMethod(r.method);
    const round = normalizeRound(r.round);

    wcData[wc].total++;
    overall.total++;

    if (method === 'KO') { wcData[wc].ko++; overall.ko++; }
    if (method === 'SUB') { wcData[wc].sub++; overall.sub++; }
    if (method === 'DEC') { wcData[wc].dec++; overall.dec++; }

    if (round === 'R1') { wcData[wc].r1++; overall.r1++; }
    if (round === 'R2') { wcData[wc].r2++; overall.r2++; }
    if (round === 'R3') { wcData[wc].r3++; overall.r3++; }
    if (round === 'R4') { wcData[wc].r4++; overall.r4++; }
    if (round === 'R5') { wcData[wc].r5++; overall.r5++; }

    wcData[wc].fights.push({
        fight: `${fight.fighterA} vs ${fight.fighterB}`,
        winner: r.winner,
        method,
        round,
        event: r.event.replace('UFC Fight Night: ', 'FN: ')
    });
});

// Weight class order (heaviest to lightest)
const wcOrder = ['HW', 'LHW', 'MW', 'WW', 'LW', 'FW', 'BW', 'FLW', 'WSW', 'WFLW', 'WBW', 'WFW'];

console.log('='.repeat(100));
console.log('WEIGHT CLASS vs METHOD/ROUND ANALYSIS - 37 Actual Results');
console.log('='.repeat(100));

console.log('\n--- Method Distribution by Weight Class ---');
console.log(`${'WC'.padEnd(6)} ${'N'.padEnd(4)} ${'KO'.padEnd(12)} ${'SUB'.padEnd(12)} ${'DEC'.padEnd(12)} ${'Finish%'.padEnd(10)}`);
console.log('─'.repeat(60));

wcOrder.forEach(wc => {
    const d = wcData[wc];
    if (!d) return;
    const finishPct = ((d.ko + d.sub) / d.total * 100).toFixed(0);
    console.log(`${wc.padEnd(6)} ${String(d.total).padEnd(4)} ${(d.ko + '/' + d.total + ' (' + (d.ko/d.total*100).toFixed(0) + '%)').padEnd(12)} ${(d.sub + '/' + d.total + ' (' + (d.sub/d.total*100).toFixed(0) + '%)').padEnd(12)} ${(d.dec + '/' + d.total + ' (' + (d.dec/d.total*100).toFixed(0) + '%)').padEnd(12)} ${finishPct}%`);
});

const overallFinish = ((overall.ko + overall.sub) / overall.total * 100).toFixed(0);
console.log('─'.repeat(60));
console.log(`${'ALL'.padEnd(6)} ${String(overall.total).padEnd(4)} ${(overall.ko + '/' + overall.total + ' (' + (overall.ko/overall.total*100).toFixed(0) + '%)').padEnd(12)} ${(overall.sub + '/' + overall.total + ' (' + (overall.sub/overall.total*100).toFixed(0) + '%)').padEnd(12)} ${(overall.dec + '/' + overall.total + ' (' + (overall.dec/overall.total*100).toFixed(0) + '%)').padEnd(12)} ${overallFinish}%`);

console.log('\n--- Round Distribution for Finishes ---');
const finishes = [];
results.forEach(r => {
    const fight = fightMap[r.event + '|' + r.winner];
    if (!fight) return;
    const method = normalizeMethod(r.method);
    if (method === 'DEC') return;
    const round = normalizeRound(r.round);
    finishes.push({ wc: fight.weightClass, method, round, fight: `${fight.fighterA} vs ${fight.fighterB}`, winner: r.winner });
});

console.log(`\nTotal finishes: ${finishes.length}/${overall.total} (${(finishes.length/overall.total*100).toFixed(0)}%)`);
console.log(`  R1: ${overall.r1} (${(overall.r1/finishes.length*100).toFixed(0)}% of finishes)`);
console.log(`  R2: ${overall.r2} (${(overall.r2/finishes.length*100).toFixed(0)}% of finishes)`);
console.log(`  R3: ${overall.r3} (${(overall.r3/finishes.length*100).toFixed(0)}% of finishes)`);
if (overall.r4 + overall.r5 > 0) console.log(`  R4+: ${overall.r4 + overall.r5}`);

console.log('\n--- Every Finish (by weight class) ---');
wcOrder.forEach(wc => {
    const wcFinishes = finishes.filter(f => f.wc === wc);
    if (wcFinishes.length === 0) return;
    console.log(`\n  ${wc}:`);
    wcFinishes.forEach(f => {
        console.log(`    ${f.method} ${f.round}  ${f.winner.padEnd(25)} ${f.fight}`);
    });
});

// Prediction vs actual comparison for our model
console.log('\n\n--- What We Predicted vs Actual (Finishes Only) ---');
console.log('Shows whether model predicted KO/SUB correctly for actual finishes\n');

const predMap = {};
predictions.forEach(p => {
    const fight = fightMap[p.event + '|' + p.winner];
    if (!fight) return;
    predMap[p.event + '|' + fight.fighterA] = p;
    predMap[p.event + '|' + fight.fighterB] = p;
});

finishes.forEach(f => {
    const pred = predMap[f.fight.split(' vs ')[0] ? f.wc : null]; // won't work perfectly
});

// Simpler: compare by event + winner
console.log(`${'WC'.padEnd(5)} ${'Actual'.padEnd(15)} ${'Predicted'.padEnd(15)} ${'W?'.padEnd(4)} ${'M?'.padEnd(4)} Fight`);
results.forEach(r => {
    const fight = fightMap[r.event + '|' + r.winner];
    if (!fight) return;
    const method = normalizeMethod(r.method);
    if (method === 'DEC') return;
    const round = normalizeRound(r.round);

    // Find the prediction for this fight
    const pred = predictions.find(p => p.event === r.event && (
        fightMap[p.event + '|' + p.winner] === fight
    ));
    if (!pred) return;

    const predMethod = normalizeMethod(pred.method);
    const predRound = normalizeRound(pred.round);
    const winCorrect = pred.winner === r.winner ? 'Y' : 'N';
    const methCorrect = (pred.winner === r.winner && predMethod === method) ? 'Y' : 'N';

    console.log(`${fight.weightClass.padEnd(5)} ${(method + ' ' + round).padEnd(15)} ${(predMethod + ' ' + predRound).padEnd(15)} ${winCorrect.padEnd(4)} ${methCorrect.padEnd(4)} ${fight.fighterA} vs ${fight.fighterB}`);
});
