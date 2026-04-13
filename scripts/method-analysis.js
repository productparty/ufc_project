/**
 * Method accuracy analysis from DB dump
 */
const fs = require('fs');
const path = require('path');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results', 'db-dump.json'), 'utf8'));

const results = data.results;
const preds = data.predictions;
const fights = data.fights;

const predMap = {};
preds.forEach(p => { predMap[p.fightId] = p; });

const fightMap = {};
fights.forEach(f => { fightMap[f.id] = f; });

function normM(m) {
    if (!m) return '';
    m = m.toUpperCase();
    if (['KO', 'TKO', 'KO/TKO'].includes(m)) return 'KO';
    if (['SUB', 'SUBMISSION'].includes(m)) return 'SUB';
    if (['DEC', 'DECISION', 'UD', 'SD', 'MD'].includes(m)) return 'DEC';
    return m;
}

let total = 0, wCorrect = 0;
let decPred = 0, finPred = 0, decPredMethodOk = 0, finPredMethodOk = 0;
let predDEC_actFin = 0, predFin_actDEC = 0;
let predDEC_actSUB = 0, predDEC_actKO = 0;
let actDEC = 0, actFin = 0;

const decActFinDetails = [];
const finActDecDetails = [];

results.forEach(r => {
    const p = predMap[r.fightId];
    if (!p || !r.winnerName) return;
    total++;

    const winCorrect = p.winnerName === r.winnerName;
    if (winCorrect) wCorrect++;

    const predMethod = normM(p.method);
    const actMethod = normM(r.method);

    if (actMethod === 'DEC') actDEC++;
    else actFin++;

    if (predMethod === 'DEC') {
        decPred++;
        if (winCorrect && predMethod === actMethod) decPredMethodOk++;
    } else {
        finPred++;
        if (winCorrect && predMethod === actMethod) finPredMethodOk++;
    }

    if (winCorrect && predMethod === 'DEC' && actMethod !== 'DEC') {
        predDEC_actFin++;
        const fight = fightMap[r.fightId];
        decActFinDetails.push({
            fight: fight ? `${fight.fighterA.name} vs ${fight.fighterB.name}` : r.fightId,
            wc: fight ? fight.weightClass : '?',
            predicted: `${p.winnerName} by DEC`,
            actual: `${r.winnerName} by ${actMethod} ${r.round || ''}`,
            confidence: p.confidence ? p.confidence.toFixed(1) : 'N/A'
        });
        if (actMethod === 'SUB') predDEC_actSUB++;
        if (actMethod === 'KO') predDEC_actKO++;
    }
    if (winCorrect && predMethod !== 'DEC' && actMethod === 'DEC') {
        predFin_actDEC++;
        const fight = fightMap[r.fightId];
        finActDecDetails.push({
            fight: fight ? `${fight.fighterA.name} vs ${fight.fighterB.name}` : r.fightId,
            wc: fight ? fight.weightClass : '?',
            predicted: `${p.winnerName} by ${predMethod}`,
            actual: `${r.winnerName} by DEC`,
            confidence: p.confidence ? p.confidence.toFixed(1) : 'N/A'
        });
    }
});

console.log('='.repeat(80));
console.log('METHOD ACCURACY ANALYSIS - DB DUMP (v1/v2 predictions)');
console.log('='.repeat(80));
console.log();
console.log(`Total fights scored: ${total}`);
console.log(`Winner accuracy: ${wCorrect}/${total} (${(wCorrect / total * 100).toFixed(1)}%)`);
console.log();
console.log('--- Method Distribution ---');
console.log(`Predicted DEC: ${decPred}/${total} (${(decPred / total * 100).toFixed(1)}%)`);
console.log(`Predicted Finish: ${finPred}/${total} (${(finPred / total * 100).toFixed(1)}%)`);
console.log(`Actual DEC: ${actDEC}/${total} (${(actDEC / total * 100).toFixed(1)}%)`);
console.log(`Actual Finish: ${actFin}/${total} (${(actFin / total * 100).toFixed(1)}%)`);
console.log();
console.log('--- Method Accuracy (when winner correct) ---');
console.log(`DEC predicted & correct: ${decPredMethodOk}/${decPred} (${(decPredMethodOk / decPred * 100).toFixed(1)}%)`);
console.log(`Finish predicted & correct: ${finPredMethodOk}/${finPred} (${(finPredMethodOk / finPred * 100).toFixed(1)}%)`);
console.log();
console.log('--- Method Mismatches (winner correct, method wrong) ---');
console.log(`Predicted DEC but was FINISH: ${predDEC_actFin} (${predDEC_actKO} KO, ${predDEC_actSUB} SUB) → lost ${predDEC_actFin * 3}pts`);
console.log(`Predicted FINISH but was DEC: ${predFin_actDEC} → lost ${predFin_actDEC * 3}pts`);
console.log();

if (decActFinDetails.length > 0) {
    console.log('  DEC→Finish misses (under-predicted finishes):');
    decActFinDetails.forEach(d => {
        console.log(`    [${d.wc}] ${d.fight.padEnd(42)} conf: ${d.confidence}% | ${d.predicted} → ${d.actual}`);
    });
}

if (finActDecDetails.length > 0) {
    console.log('  Finish→DEC misses (over-predicted finishes):');
    finActDecDetails.forEach(d => {
        console.log(`    [${d.wc}] ${d.fight.padEnd(42)} conf: ${d.confidence}% | ${d.predicted} → ${d.actual}`);
    });
}

// Key insight: what's the net directional error?
console.log();
console.log('='.repeat(80));
console.log('KEY INSIGHT: DEC vs FINISH BIAS');
console.log('='.repeat(80));
const decOverPredict = decPred - actDEC;
console.log(`DEC over/under-prediction: ${decOverPredict > 0 ? '+' : ''}${decOverPredict} (predicted ${decPred} DEC, actual ${actDEC})`);
console.log(`Finish over/under-prediction: ${-decOverPredict > 0 ? '+' : ''}${-decOverPredict}`);
if (decOverPredict > 0) {
    console.log(`→ Model is OVER-predicting DEC by ${decOverPredict} fights`);
} else if (decOverPredict < 0) {
    console.log(`→ Model is UNDER-predicting DEC by ${Math.abs(decOverPredict)} fights`);
} else {
    console.log(`→ Model has PERFECT DEC/Finish distribution balance`);
}
