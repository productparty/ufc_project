/**
 * UFC Predictor Backtest Script
 * Scores old predictions against actual results using the scoring system:
 * - 5pts correct winner
 * - +3pts correct method (only if winner correct)
 * - +2pts correct round (only if method correct, finishes only)
 * DEC max = 8pts, Finish max = 10pts
 */

const fs = require('fs');

// Load the saved data
const data = JSON.parse(fs.readFileSync('results/ufc-predictor-2026-02-08T04-07-22-367Z.json', 'utf8'));

const predictions = data.allPredictions;
const results = data.allResults;
const fights = data.allFights;

// Normalize method for comparison
function normalizeMethod(method) {
    if (!method) return '';
    const m = method.toUpperCase().trim();
    if (['KO', 'TKO', 'KO/TKO'].includes(m)) return 'KO';
    if (['SUB', 'SUBMISSION'].includes(m)) return 'SUB';
    if (['DEC', 'DECISION', 'UD', 'SD', 'MD'].includes(m)) return 'DEC';
    return m;
}

// Normalize round for comparison
function normalizeRound(round) {
    if (!round) return '';
    const r = String(round).toUpperCase().trim();
    if (r === 'DEC' || r === 'DECISION') return 'DEC';
    // Handle "R1" -> "R1", "1" -> "R1"
    if (r.startsWith('R')) return r;
    if (!isNaN(parseInt(r))) return 'R' + r;
    return r;
}

// Build lookup: event+winner -> result
const resultMap = {};
results.forEach(r => {
    const key = r.event + '|' + r.winner;
    resultMap[key] = r;
});

// Build fight lookup: event+fighterA+fighterB -> fight info
const fightMap = {};
fights.forEach(f => {
    fightMap[f.event + '|' + f.fighterA] = f;
    fightMap[f.event + '|' + f.fighterB] = f;
});

// Score each prediction
const events = {};
let totalPoints = 0;
let maxPossiblePoints = 0;
let totalFights = 0;
let winnerCorrect = 0;
let methodCorrect = 0;
let roundCorrect = 0;
let finishPredictions = 0;
let decPredictions = 0;
let actualFinishes = 0;
let actualDecs = 0;

// Find matching result for each prediction
predictions.forEach(pred => {
    // Find the result for this fight
    const fight = fightMap[pred.event + '|' + pred.winner];
    if (!fight) return;

    const fighterA = fight.fighterA;
    const fighterB = fight.fighterB;

    // Find result by checking both fighters
    const resultA = resultMap[pred.event + '|' + fighterA];
    const resultB = resultMap[pred.event + '|' + fighterB];
    const result = resultA || resultB;
    if (!result) return;

    totalFights++;
    if (!events[pred.event]) {
        events[pred.event] = { fights: [], points: 0, maxPoints: 0, winnerCorrect: 0, methodCorrect: 0, roundCorrect: 0, total: 0 };
    }

    const predMethod = normalizeMethod(pred.method);
    const actualMethod = normalizeMethod(result.method);
    const predRound = normalizeRound(pred.round);
    const actualRound = normalizeRound(result.round);

    // Track finish vs DEC distribution
    if (predMethod !== 'DEC') finishPredictions++;
    else decPredictions++;
    if (actualMethod !== 'DEC') actualFinishes++;
    else actualDecs++;

    let points = 0;
    let wCorrect = false, mCorrect = false, rCorrect = false;

    // Max possible: DEC = 8, Finish = 10
    const maxForFight = actualMethod === 'DEC' ? 8 : 10;
    maxPossiblePoints += maxForFight;

    // Winner correct?
    if (pred.winner === result.winner) {
        points += 5;
        wCorrect = true;
        winnerCorrect++;

        // Method correct?
        if (predMethod === actualMethod) {
            points += 3;
            mCorrect = true;
            methodCorrect++;

            // Round correct? (only for finishes)
            if (actualMethod !== 'DEC') {
                if (predRound === actualRound) {
                    points += 2;
                    rCorrect = true;
                    roundCorrect++;
                }
            }
        }
    }

    totalPoints += points;
    events[pred.event].points += points;
    events[pred.event].maxPoints += maxForFight;
    events[pred.event].total++;
    if (wCorrect) events[pred.event].winnerCorrect++;
    if (mCorrect) events[pred.event].methodCorrect++;
    if (rCorrect) events[pred.event].roundCorrect++;

    events[pred.event].fights.push({
        fight: `${fighterA} vs ${fighterB}`,
        wc: fight.weightClass,
        predicted: `${pred.winner} by ${predMethod} ${predRound}`,
        actual: `${result.winner} by ${actualMethod} ${actualRound}`,
        winner: wCorrect ? 'Y' : 'N',
        method: mCorrect ? 'Y' : '-',
        round: rCorrect ? 'Y' : '-',
        points: points,
        maxPts: maxForFight,
        confidence: pred.confidence?.toFixed(1) || 'N/A'
    });
});

// Print results
console.log('='.repeat(100));
console.log('UFC PREDICTOR BACKTEST RESULTS (Old Model v1)');
console.log('='.repeat(100));
console.log();

Object.entries(events).forEach(([eventName, ev]) => {
    console.log(`\n--- ${eventName} ---`);
    console.log(`Winner: ${ev.winnerCorrect}/${ev.total} (${(ev.winnerCorrect/ev.total*100).toFixed(1)}%) | Method: ${ev.methodCorrect}/${ev.total} | Round: ${ev.roundCorrect}/${ev.total}`);
    console.log(`Points: ${ev.points}/${ev.maxPoints} (${(ev.points/ev.maxPoints*100).toFixed(1)}%)`);
    console.log();

    ev.fights.forEach(f => {
        const status = f.winner === 'Y' ? (f.method === 'Y' ? (f.round === 'Y' ? 'PERFECT' : 'W+M') : 'W') : 'MISS';
        console.log(`  ${status.padEnd(8)} ${f.points}/${f.maxPts}pts  ${f.fight.padEnd(45)} P: ${f.predicted.padEnd(25)} A: ${f.actual.padEnd(20)} conf: ${f.confidence}%`);
    });
});

console.log('\n' + '='.repeat(100));
console.log('OVERALL SUMMARY');
console.log('='.repeat(100));
console.log(`Total Fights Scored: ${totalFights}`);
console.log(`Winner Accuracy:  ${winnerCorrect}/${totalFights} (${(winnerCorrect/totalFights*100).toFixed(1)}%)`);
console.log(`Method Accuracy:  ${methodCorrect}/${totalFights} (${(methodCorrect/totalFights*100).toFixed(1)}%)`);
console.log(`Round Accuracy:   ${roundCorrect}/${totalFights} (${(roundCorrect/totalFights*100).toFixed(1)}%)`);
console.log(`Total Points:     ${totalPoints}/${maxPossiblePoints} (${(totalPoints/maxPossiblePoints*100).toFixed(1)}%)`);
console.log();
console.log('--- Method Distribution ---');
console.log(`Predicted Finishes: ${finishPredictions}/${totalFights} (${(finishPredictions/totalFights*100).toFixed(1)}%)`);
console.log(`Predicted DECs:     ${decPredictions}/${totalFights} (${(decPredictions/totalFights*100).toFixed(1)}%)`);
console.log(`Actual Finishes:    ${actualFinishes}/${totalFights} (${(actualFinishes/totalFights*100).toFixed(1)}%)`);
console.log(`Actual DECs:        ${actualDecs}/${totalFights} (${(actualDecs/totalFights*100).toFixed(1)}%)`);

// Identify where method was wrong but winner was right (where v3 changes matter)
console.log('\n--- Method Mismatches (Winner Correct, Method Wrong) ---');
console.log('These are fights where source reweighting / finish calibration would help:');
Object.entries(events).forEach(([eventName, ev]) => {
    ev.fights.filter(f => f.winner === 'Y' && f.method !== 'Y').forEach(f => {
        console.log(`  ${f.fight.padEnd(45)} Predicted: ${f.predicted.padEnd(25)} Actual: ${f.actual}`);
    });
});
