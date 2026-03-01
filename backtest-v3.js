/**
 * UFC Predictor Backtest: v1 (old predictions) vs v3 (current engine)
 * Re-runs the v3 prediction engine on historical fight data from the DB dump,
 * then compares both against actual results.
 *
 * Scoring: 5pts winner, +3pts method (if winner correct), +2pts round (if method correct, finishes only)
 * DEC max = 8pts, Finish max = 10pts
 */

const fs = require('fs');

// Load the DB dump (has full fighter stat objects)
const dump = JSON.parse(fs.readFileSync('results/db-dump.json', 'utf8'));

// Load prediction engine
const engineCode = fs.readFileSync('prediction-engine.js', 'utf8');
const PredictionEngine = new Function(engineCode + '\nreturn PredictionEngine;')();
const engine = new PredictionEngine();

// Build event ID -> name map
const eventMap = {};
dump.events.forEach(e => { eventMap[e.id] = e; });

// Build fight ID -> fight map
const fightMap = {};
dump.fights.forEach(f => { fightMap[f.id] = f; });

// Build results by fightId
const resultMap = {};
dump.results.forEach(r => { resultMap[r.fightId] = r; });

// Build old predictions by fightId
const oldPredMap = {};
dump.predictions.forEach(p => { oldPredMap[p.fightId] = p; });

// Normalize method for comparison
function normalizeMethod(method) {
    if (!method) return '';
    const m = method.toUpperCase().trim();
    if (['KO', 'TKO', 'KO/TKO'].includes(m)) return 'KO';
    if (['SUB', 'SUBMISSION'].includes(m)) return 'SUB';
    if (['DEC', 'DECISION', 'UD', 'SD', 'MD'].includes(m)) return 'DEC';
    return m;
}

// Normalize round
function normalizeRound(round) {
    if (!round) return '';
    const r = String(round).toUpperCase().trim();
    if (r === 'DEC' || r === 'DECISION') return 'DEC';
    if (r.startsWith('R')) return r;
    if (!isNaN(parseInt(r))) return 'R' + r;
    return r;
}

// Score a prediction against a result
function scorePrediction(pred, result) {
    const predMethod = normalizeMethod(pred.method);
    const actualMethod = normalizeMethod(result.method);
    const predRound = normalizeRound(pred.round);
    const actualRound = normalizeRound(result.round);
    const maxPts = actualMethod === 'DEC' ? 8 : 10;

    let points = 0;
    let winner = false, method = false, round = false;

    if (pred.winnerName === result.winnerName) {
        points += 5;
        winner = true;
        if (predMethod === actualMethod) {
            points += 3;
            method = true;
            if (actualMethod !== 'DEC' && predRound === actualRound) {
                points += 2;
                round = true;
            }
        }
    }

    return { points, maxPts, winner, method, round, predMethod, actualMethod, predRound, actualRound };
}

// Find events with both fighter data AND results
const testableEvents = [];
dump.events.forEach(ev => {
    const eventFights = dump.fights.filter(f => f.eventId === ev.id);
    const hasData = eventFights.some(f => f.fighterA && f.fighterA.tapology && f.fighterA.tapology.consensus !== null);
    const hasResults = dump.results.some(r => r.eventId === ev.id);
    if (hasData && hasResults) {
        testableEvents.push(ev);
    }
});

console.log('='.repeat(110));
console.log('UFC PREDICTOR BACKTEST: v1 (Old) vs v3 (Current Engine)');
console.log('='.repeat(110));
console.log(`Testable events: ${testableEvents.map(e => e.name).join(', ')}`);
console.log();

// Totals
let v1Total = { points: 0, maxPts: 0, fights: 0, winner: 0, method: 0, round: 0, finishPred: 0, decPred: 0 };
let v3Total = { points: 0, maxPts: 0, fights: 0, winner: 0, method: 0, round: 0, finishPred: 0, decPred: 0 };
let actualFinishes = 0, actualDecs = 0;

testableEvents.forEach(ev => {
    const eventFights = dump.fights.filter(f => f.eventId === ev.id);
    const eventResults = dump.results.filter(r => r.eventId === ev.id);

    // Determine event type
    const eventType = ev.type || 'fight-night';

    // Re-run v3 predictions
    const v3Predictions = engine.generatePredictions(eventFights, eventType);

    console.log(`\n${'─'.repeat(110)}`);
    console.log(`  ${ev.name} (${eventType})`);
    console.log(`${'─'.repeat(110)}`);
    console.log(`${'Fight'.padEnd(42)} ${'Actual'.padEnd(20)} | ${'v1 Pred'.padEnd(20)} ${'Score'.padEnd(8)} | ${'v3 Pred'.padEnd(20)} ${'Score'.padEnd(8)} | Delta`);
    console.log(`${''.padEnd(42)} ${''.padEnd(20)} | ${''.padEnd(20)} ${''.padEnd(8)} | ${''.padEnd(20)} ${''.padEnd(8)} |`);

    let evV1 = { points: 0, maxPts: 0, fights: 0, winner: 0, method: 0 };
    let evV3 = { points: 0, maxPts: 0, fights: 0, winner: 0, method: 0 };

    eventFights.forEach((fight, idx) => {
        const result = resultMap[fight.id];
        if (!result) return;

        const oldPred = oldPredMap[fight.id];
        const v3Pred = v3Predictions[idx];

        if (!oldPred || !v3Pred) return;

        const actualMethod = normalizeMethod(result.method);
        const actualRound = normalizeRound(result.round);
        const actualStr = `${result.winnerName.split(' ').pop()} ${actualMethod} ${actualRound}`;

        // Track actual distribution
        if (actualMethod !== 'DEC') actualFinishes++;
        else actualDecs++;

        // Score v1
        const v1Score = scorePrediction(oldPred, result);
        const v1Str = `${normalizeMethod(oldPred.method)} ${normalizeRound(oldPred.round)}`;

        // Score v3
        const v3Score = scorePrediction(v3Pred, result);
        const v3Str = `${normalizeMethod(v3Pred.method)} ${normalizeRound(v3Pred.round)}`;

        // Track distribution
        if (v1Score.predMethod !== 'DEC') v1Total.finishPred++;
        else v1Total.decPred++;
        if (v3Score.predMethod !== 'DEC') v3Total.finishPred++;
        else v3Total.decPred++;

        // Accumulate
        v1Total.points += v1Score.points; v1Total.maxPts += v1Score.maxPts; v1Total.fights++;
        if (v1Score.winner) v1Total.winner++;
        if (v1Score.method) v1Total.method++;
        if (v1Score.round) v1Total.round++;

        v3Total.points += v3Score.points; v3Total.maxPts += v3Score.maxPts; v3Total.fights++;
        if (v3Score.winner) v3Total.winner++;
        if (v3Score.method) v3Total.method++;
        if (v3Score.round) v3Total.round++;

        evV1.points += v1Score.points; evV1.maxPts += v1Score.maxPts; evV1.fights++;
        if (v1Score.winner) evV1.winner++;
        if (v1Score.method) evV1.method++;
        evV3.points += v3Score.points; evV3.maxPts += v3Score.maxPts; evV3.fights++;
        if (v3Score.winner) evV3.winner++;
        if (v3Score.method) evV3.method++;

        const delta = v3Score.points - v1Score.points;
        const deltaStr = delta > 0 ? `+${delta}` : delta < 0 ? `${delta}` : '=';
        const deltaColor = delta > 0 ? '>>>' : delta < 0 ? '<<<' : '   ';

        const fightName = `${fight.fighterA.name} vs ${fight.fighterB.name}`;
        const v1Status = v1Score.winner ? (v1Score.method ? 'W+M' : 'W') : 'MISS';
        const v3Status = v3Score.winner ? (v3Score.method ? 'W+M' : 'W') : 'MISS';

        console.log(`${fightName.padEnd(42)} ${actualStr.padEnd(20)} | ${v1Status.padEnd(4)} ${v1Str.padEnd(15)} ${String(v1Score.points + '/' + v1Score.maxPts).padEnd(8)} | ${v3Status.padEnd(4)} ${v3Str.padEnd(15)} ${String(v3Score.points + '/' + v3Score.maxPts).padEnd(8)} | ${deltaColor} ${deltaStr}`);
    });

    console.log();
    console.log(`  Event Summary:  v1: ${evV1.points}/${evV1.maxPts} (${(evV1.points/evV1.maxPts*100).toFixed(1)}%) W:${evV1.winner}/${evV1.fights} M:${evV1.method}/${evV1.fights}  |  v3: ${evV3.points}/${evV3.maxPts} (${(evV3.points/evV3.maxPts*100).toFixed(1)}%) W:${evV3.winner}/${evV3.fights} M:${evV3.method}/${evV3.fights}`);
});

// Overall comparison
console.log('\n' + '='.repeat(110));
console.log('OVERALL COMPARISON');
console.log('='.repeat(110));

const pctV1 = (v1Total.points / v1Total.maxPts * 100).toFixed(1);
const pctV3 = (v3Total.points / v3Total.maxPts * 100).toFixed(1);
const improvement = (v3Total.points - v1Total.points);

console.log(`\n                        v1 (Old Model)       v3 (Current)         Delta`);
console.log(`  ${'─'.repeat(80)}`);
console.log(`  Total Fights:         ${String(v1Total.fights).padEnd(20)} ${String(v3Total.fights).padEnd(20)}`);
console.log(`  Total Points:         ${(v1Total.points + '/' + v1Total.maxPts + ' (' + pctV1 + '%)').padEnd(20)} ${(v3Total.points + '/' + v3Total.maxPts + ' (' + pctV3 + '%)').padEnd(20)} ${improvement > 0 ? '+' : ''}${improvement} pts`);
console.log(`  Winner Accuracy:      ${(v1Total.winner + '/' + v1Total.fights + ' (' + (v1Total.winner/v1Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${(v3Total.winner + '/' + v3Total.fights + ' (' + (v3Total.winner/v3Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${v3Total.winner - v1Total.winner > 0 ? '+' : ''}${v3Total.winner - v1Total.winner}`);
console.log(`  Method Accuracy:      ${(v1Total.method + '/' + v1Total.fights + ' (' + (v1Total.method/v1Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${(v3Total.method + '/' + v3Total.fights + ' (' + (v3Total.method/v3Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${v3Total.method - v1Total.method > 0 ? '+' : ''}${v3Total.method - v1Total.method}`);
console.log(`  Round Accuracy:       ${(v1Total.round + '/' + v1Total.fights + ' (' + (v1Total.round/v1Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${(v3Total.round + '/' + v3Total.fights + ' (' + (v3Total.round/v3Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${v3Total.round - v1Total.round > 0 ? '+' : ''}${v3Total.round - v1Total.round}`);
console.log();
console.log(`  --- Method Distribution ---`);
console.log(`  Predicted Finishes:   ${(v1Total.finishPred + '/' + v1Total.fights + ' (' + (v1Total.finishPred/v1Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${(v3Total.finishPred + '/' + v3Total.fights + ' (' + (v3Total.finishPred/v3Total.fights*100).toFixed(1) + '%)').padEnd(20)}`);
console.log(`  Predicted DECs:       ${(v1Total.decPred + '/' + v1Total.fights + ' (' + (v1Total.decPred/v1Total.fights*100).toFixed(1) + '%)').padEnd(20)} ${(v3Total.decPred + '/' + v3Total.fights + ' (' + (v3Total.decPred/v3Total.fights*100).toFixed(1) + '%)').padEnd(20)}`);
console.log(`  Actual Finishes:      ${actualFinishes}/${v1Total.fights} (${(actualFinishes/v1Total.fights*100).toFixed(1)}%)`);
console.log(`  Actual DECs:          ${actualDecs}/${v1Total.fights} (${(actualDecs/v1Total.fights*100).toFixed(1)}%)`);

// Show per-fight delta details for changes
console.log('\n' + '='.repeat(110));
console.log('FIGHTS WHERE v3 DIFFERED FROM v1');
console.log('='.repeat(110));

testableEvents.forEach(ev => {
    const eventFights = dump.fights.filter(f => f.eventId === ev.id);
    const eventType = ev.type || 'fight-night';
    const v3Predictions = engine.generatePredictions(eventFights, eventType);

    eventFights.forEach((fight, idx) => {
        const result = resultMap[fight.id];
        if (!result) return;
        const oldPred = oldPredMap[fight.id];
        const v3Pred = v3Predictions[idx];
        if (!oldPred || !v3Pred) return;

        const v1Method = normalizeMethod(oldPred.method);
        const v3Method = normalizeMethod(v3Pred.method);
        const v1Winner = oldPred.winnerName;
        const v3Winner = v3Pred.winnerName;

        if (v1Method !== v3Method || v1Winner !== v3Winner) {
            const v1Score = scorePrediction(oldPred, result);
            const v3Score = scorePrediction(v3Pred, result);
            const delta = v3Score.points - v1Score.points;
            const icon = delta > 0 ? 'IMPROVED' : delta < 0 ? 'REGRESSED' : 'SAME PTS';

            const fightName = `${fight.fighterA.name} vs ${fight.fighterB.name}`;
            console.log(`\n  [${icon}] ${fightName} (${fight.weightClass})`);
            console.log(`    Actual:  ${result.winnerName} by ${normalizeMethod(result.method)} ${normalizeRound(result.round)}`);
            console.log(`    v1:      ${v1Winner} by ${v1Method} ${normalizeRound(oldPred.round)}  → ${v1Score.points}/${v1Score.maxPts}pts`);
            console.log(`    v3:      ${v3Winner} by ${v3Method} ${normalizeRound(v3Pred.round)}  → ${v3Score.points}/${v3Score.maxPts}pts  (${delta > 0 ? '+' : ''}${delta})`);

            // Show v3 reasoning highlights
            if (v3Pred.reasoning) {
                const methodReasons = v3Pred.reasoning.method || [];
                const keyReasons = methodReasons.filter(r =>
                    r.note && (r.note.includes('Close fight') || r.note.includes('EV threshold') ||
                    r.note.includes('finish') || r.note.includes('DEC') || r.note.includes('striking'))
                );
                if (keyReasons.length > 0) {
                    console.log(`    v3 reasoning: ${keyReasons.map(r => r.note).join(' | ')}`);
                }
            }
        }
    });
});
