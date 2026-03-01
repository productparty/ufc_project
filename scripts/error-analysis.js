/**
 * Deep error analysis across all scored events
 * Identifies systematic patterns in prediction failures
 */

const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

// Load the old export (has v1 predictions for 324/325, v2 for Bautista, plus all results)
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'results', 'ufc-predictor-2026-02-08T04-07-22-367Z.json'), 'utf8'));

const predictions = data.allPredictions;
const results = data.allResults;
const fights = data.allFights;

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

// Build lookups
const resultMap = {};
results.forEach(r => { resultMap[r.event + '|' + r.winner] = r; });

const fightMap = {};
fights.forEach(f => {
    fightMap[f.event + '|' + f.fighterA] = f;
    fightMap[f.event + '|' + f.fighterB] = f;
});

// Categorize all errors
const errors = {
    winnerWrong: [],          // Got the winner completely wrong
    methodWrong_predKO_actDEC: [],   // Predicted KO, was DEC
    methodWrong_predDEC_actKO: [],   // Predicted DEC, was KO
    methodWrong_predKO_actSUB: [],   // Predicted KO, was SUB
    methodWrong_predSUB_actKO: [],   // Predicted SUB, was KO
    methodWrong_predDEC_actSUB: [],  // Predicted DEC, was SUB
    methodWrong_predSUB_actDEC: [],  // Predicted SUB, was DEC
    roundWrong: [],           // Right winner+method, wrong round
    perfect: []               // Perfect predictions
};

// Track by weight class and confidence
const wcStats = {};
const confBuckets = { low: {w:0,t:0}, med: {w:0,t:0}, high: {w:0,t:0} };
const favUnderdog = { bigFav: {w:0,t:0}, slight: {w:0,t:0}, tossup: {w:0,t:0} };

predictions.forEach(pred => {
    const fight = fightMap[pred.event + '|' + pred.winner];
    if (!fight) return;

    const fighterA = fight.fighterA;
    const fighterB = fight.fighterB;
    const resultA = resultMap[pred.event + '|' + fighterA];
    const resultB = resultMap[pred.event + '|' + fighterB];
    const result = resultA || resultB;
    if (!result) return;

    const predMethod = normalizeMethod(pred.method);
    const actualMethod = normalizeMethod(result.method);
    const predRound = normalizeRound(pred.round);
    const actualRound = normalizeRound(result.round);
    const conf = pred.confidence || 50;
    const wc = fight.weightClass || 'UNK';

    const fightInfo = {
        fight: `${fighterA} vs ${fighterB}`,
        event: pred.event.replace('UFC Fight Night: ', 'FN: '),
        wc,
        conf: conf.toFixed(1),
        predicted: `${pred.winner.split(' ').pop()} by ${predMethod} ${predRound}`,
        actual: `${result.winner.split(' ').pop()} by ${actualMethod} ${actualRound}`,
        predWinner: pred.winner,
        actWinner: result.winner,
        predMethod,
        actualMethod,
        predRound,
        actualRound
    };

    // Weight class tracking
    if (!wcStats[wc]) wcStats[wc] = { total: 0, winnerRight: 0, methodRight: 0, predFinish: 0, actFinish: 0 };
    wcStats[wc].total++;
    if (predMethod !== 'DEC') wcStats[wc].predFinish++;
    if (actualMethod !== 'DEC') wcStats[wc].actFinish++;

    // Confidence bucket
    const bucket = conf >= 75 ? 'high' : conf >= 60 ? 'med' : 'low';
    confBuckets[bucket].t++;

    // Favorite bucket
    const favBucket = conf >= 75 ? 'bigFav' : conf >= 60 ? 'slight' : 'tossup';
    favUnderdog[favBucket].t++;

    const winnerCorrect = pred.winner === result.winner;

    if (winnerCorrect) {
        wcStats[wc].winnerRight++;
        confBuckets[bucket].w++;
        favUnderdog[favBucket].w++;

        if (predMethod === actualMethod) {
            wcStats[wc].methodRight++;
            if (actualMethod !== 'DEC' && predRound === actualRound) {
                errors.perfect.push(fightInfo);
            } else if (actualMethod !== 'DEC') {
                errors.roundWrong.push(fightInfo);
            }
        } else {
            // Method mismatch categories
            const key = `methodWrong_pred${predMethod}_act${actualMethod}`;
            if (errors[key]) errors[key].push(fightInfo);
        }
    } else {
        errors.winnerWrong.push(fightInfo);
    }
});

// Print analysis
console.log('='.repeat(100));
console.log('ERROR PATTERN ANALYSIS - All 37 Scored Fights (v1: 324/325, v2: Bautista)');
console.log('='.repeat(100));

console.log('\n--- WINNER WRONG (10 fights) ---');
console.log('These cost 0pts each. Fixing any = +5 to +10pts.\n');
errors.winnerWrong.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%`);
    console.log(`         Predicted: ${e.predicted.padEnd(25)} Actual: ${e.actual}`);
});

console.log('\n--- METHOD WRONG: Predicted KO, Actual DEC (common over-prediction) ---');
errors.methodWrong_predKO_actDEC.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- METHOD WRONG: Predicted DEC, Actual KO (under-prediction of finish) ---');
errors.methodWrong_predDEC_actKO.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- METHOD WRONG: Predicted KO, Actual SUB ---');
errors.methodWrong_predKO_actSUB.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- METHOD WRONG: Predicted SUB, Actual KO ---');
errors.methodWrong_predSUB_actKO.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- METHOD WRONG: Predicted DEC, Actual SUB ---');
errors.methodWrong_predDEC_actSUB.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- METHOD WRONG: Predicted SUB, Actual DEC ---');
errors.methodWrong_predSUB_actDEC.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} conf: ${e.conf}%  ${e.event}`);
});

console.log('\n--- ROUND WRONG (right winner + method, wrong round) ---');
errors.roundWrong.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} P: ${e.predRound} A: ${e.actualRound}  conf: ${e.conf}%`);
});

console.log('\n--- PERFECT (10/10 or 8/8) ---');
errors.perfect.forEach(e => {
    console.log(`  [${e.wc}] ${e.fight.padEnd(45)} ${e.predicted}  conf: ${e.conf}%`);
});

// Summary statistics
console.log('\n' + '='.repeat(100));
console.log('PATTERN SUMMARY');
console.log('='.repeat(100));

console.log('\n--- Winner Accuracy by Confidence ---');
Object.entries(confBuckets).forEach(([bucket, {w, t}]) => {
    const label = bucket === 'high' ? '75%+' : bucket === 'med' ? '60-74%' : '<60%';
    console.log(`  ${label.padEnd(10)} ${w}/${t} (${t > 0 ? (w/t*100).toFixed(1) : 0}%)`);
});

console.log('\n--- Winner Accuracy by Weight Class ---');
Object.entries(wcStats)
    .sort((a, b) => b[1].total - a[1].total)
    .forEach(([wc, s]) => {
        console.log(`  ${wc.padEnd(5)} W: ${s.winnerRight}/${s.total} (${(s.winnerRight/s.total*100).toFixed(0)}%)  M: ${s.methodRight}/${s.total}  PredFinish: ${s.predFinish}/${s.total}  ActFinish: ${s.actFinish}/${s.total}`);
    });

console.log('\n--- Error Type Counts ---');
console.log(`  Winner wrong:           ${errors.winnerWrong.length}/37 (${(errors.winnerWrong.length/37*100).toFixed(1)}%)`);
const methodWrongTotal = errors.methodWrong_predKO_actDEC.length + errors.methodWrong_predDEC_actKO.length +
    errors.methodWrong_predKO_actSUB.length + errors.methodWrong_predSUB_actKO.length +
    errors.methodWrong_predDEC_actSUB.length + errors.methodWrong_predSUB_actDEC.length;
console.log(`  Method wrong (W right): ${methodWrongTotal}/37 (${(methodWrongTotal/37*100).toFixed(1)}%)`);
console.log(`    KO predicted, was DEC:  ${errors.methodWrong_predKO_actDEC.length}  ← v3 close-fight DEC gravity targets this`);
console.log(`    DEC predicted, was KO:  ${errors.methodWrong_predDEC_actKO.length}  ← under-predicting finishes`);
console.log(`    KO predicted, was SUB:  ${errors.methodWrong_predKO_actSUB.length}  ← wrong finish type`);
console.log(`    SUB predicted, was KO:  ${errors.methodWrong_predSUB_actKO.length}  ← wrong finish type`);
console.log(`    DEC predicted, was SUB: ${errors.methodWrong_predDEC_actSUB.length}  ← under-predicting finishes`);
console.log(`    SUB predicted, was DEC: ${errors.methodWrong_predSUB_actDEC.length}  ← over-predicting SUB`);
console.log(`  Round wrong:            ${errors.roundWrong.length}/37`);
console.log(`  Perfect:                ${errors.perfect.length}/37`);

// Biggest opportunity analysis
console.log('\n--- BIGGEST POINT OPPORTUNITIES ---');
console.log('Each winner fix = +5 to +10pts. Each method fix = +3pts.\n');

const winnerWrongHighConf = errors.winnerWrong.filter(e => parseFloat(e.conf) >= 70);
console.log(`  High-confidence upsets (conf >= 70%): ${winnerWrongHighConf.length}`);
winnerWrongHighConf.forEach(e => {
    console.log(`    [${e.wc}] ${e.fight.padEnd(42)} conf: ${e.conf}%  P: ${e.predWinner.split(' ').pop()}  A: ${e.actWinner.split(' ').pop()}`);
});

const koActDec = errors.methodWrong_predKO_actDEC;
console.log(`\n  KO→DEC fixes (${koActDec.length} fights × 3pts = ${koActDec.length * 3}pts available):`);
koActDec.forEach(e => {
    console.log(`    [${e.wc}] ${e.fight.padEnd(42)} conf: ${e.conf}%`);
});
