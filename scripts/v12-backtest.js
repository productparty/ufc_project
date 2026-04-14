/**
 * v12 Parameter Backtest
 *
 * Tests proposed v12 changes against all 103 historical fights to measure impact.
 * Uses export JSON (predictions + results) to simulate what WOULD have changed.
 *
 * Key question: Which heuristic rules fix the most method misses with fewest regressions?
 *
 * Proposed v12 changes to test:
 * 1. High-confidence DEC regression: confidence >= 75% + KO predicted + non-HW → switch to DEC
 * 2. Raise DEC_EV_MULTIPLIER from 1.25 → 1.30/1.35
 * 3. Cap cumulative finish multipliers
 *
 * Since we can't re-run the full engine (no raw fighter data in export), we test
 * post-hoc rules against prediction metadata (confidence, method, weightClass, source).
 */
const fs = require('fs');
const path = require('path');

// Find latest export
const projectDir = path.join(__dirname, '..');
const exportFiles = fs.readdirSync(projectDir)
    .filter(f => f.startsWith('ufc-predictor-') && f.endsWith('.json'))
    .sort()
    .reverse();

if (exportFiles.length === 0) {
    console.error('No export files found');
    process.exit(1);
}

const exportFile = path.join(projectDir, exportFiles[0]);
console.log('Using:', exportFiles[0]);
const data = JSON.parse(fs.readFileSync(exportFile, 'utf8'));

const preds = data.allPredictions;
const results = data.allResults;
const fights = data.allFights;

// Build fight lookup by event + fighter names
function findFight(pred) {
    return fights.find(f =>
        f.event === pred.event &&
        (pred.winner === f.fighterA || pred.winner === f.fighterB)
    );
}

// Build result lookup by event + fight match
function findResult(fight, usedResults) {
    const eventResults = results.filter(r =>
        r.event === fight.event &&
        r.method !== 'CANCELLED' &&
        r.winner !== 'cancelled' &&
        !usedResults.has(r)
    );
    for (const r of eventResults) {
        const wLast = r.winner.split(' ').pop().toLowerCase();
        const aLast = fight.fighterA.split(' ').pop().toLowerCase();
        const bLast = fight.fighterB.split(' ').pop().toLowerCase();
        if (wLast === aLast || wLast === bLast || r.winner === fight.fighterA || r.winner === fight.fighterB) {
            usedResults.add(r);
            return r;
        }
    }
    return null;
}

// Match all predictions to fights and results
const usedResults = new Set();
const matched = [];
for (const pred of preds) {
    const fight = findFight(pred);
    if (!fight) continue;
    const result = findResult(fight, usedResults);
    if (!result) continue;

    const effWinner = pred.override ? (pred.override.winner || pred.winner) : pred.winner;
    const effMethod = pred.override ? (pred.override.method || pred.method) : pred.method;
    const effRound = pred.override ? (pred.override.round || pred.round) : pred.round;

    const winnerCorrect = result.winner.split(' ').pop().toLowerCase() === effWinner.split(' ').pop().toLowerCase()
        || result.winner === effWinner;

    matched.push({
        event: fight.event.replace('UFC ', ''),
        fighterA: fight.fighterA,
        fighterB: fight.fighterB,
        wc: fight.weightClass,
        predWinner: effWinner,
        predMethod: effMethod,
        predRound: effRound,
        confidence: pred.confidence || 0,
        source: pred.source || '',
        actualWinner: result.winner,
        actualMethod: result.method,
        actualRound: result.round,
        winnerCorrect,
        methodCorrect: winnerCorrect && effMethod === result.method,
        roundCorrect: winnerCorrect && effMethod === result.method &&
            (effMethod === 'DEC' ? true : effRound === result.round),
    });
}

console.log(`\nMatched ${matched.length} fights across ${data.events.length} events\n`);

// === SCORING ===
// League scoring (observed from UFC 327 scoreboard):
//   W=150 base + 100 Win Bonus (whenever winner correct) → 250 for winner-only
//   M=125 (gated by W)   R=50 (gated by W+non-DEC method)
// Ulberg KO R1 perfect = 150+100+125+50 = 425 ✓
// Hokit DEC R3 perfect = 150+100+125+0  = 375 (observed 275 → no Win Bonus on DEC picks?)
// Reyes KO R1 picked, Reyes DEC won = 150+100+0+0 = 250 (observed 300 → mystery 50)
// Treat scoring as approximate; use deltas between v14/v15 runs, not absolute totals.
function calcScore(winnerCorrect, methodCorrect, roundCorrect) {
    if (!winnerCorrect) return 0;
    let score = 150 + 100; // base + Win Bonus
    if (methodCorrect) score += 125;
    if (roundCorrect) score += 50;
    return score;
}

// === TEST RULES ===
// Each rule takes a fight and returns a modified method (or null to keep original)
const rules = {
    // Rule 1: High-confidence KO → DEC for non-HW (various thresholds)
    'HC-KO→DEC (conf≥80, non-HW)': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 80 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },
    'HC-KO→DEC (conf≥75, non-HW)': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 75 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },
    'HC-KO→DEC (conf≥70, non-HW)': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 70 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },
    'HC-KO→DEC (conf≥85, non-HW)': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 85 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },

    // Rule 2: Any KO in lighter divisions → DEC (aggressive)
    'KO→DEC in BW/FLW': (f) => {
        if (f.predMethod === 'KO' && ['BW', 'FLW'].includes(f.wc)) return 'DEC';
        return null;
    },

    // Rule 3: HC-finish→DEC regardless of method type
    'HC-finish→DEC (conf≥80, non-HW)': (f) => {
        if ((f.predMethod === 'KO' || f.predMethod === 'SUB') && f.confidence >= 80 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },

    // Rule 4: KO in MW/WW/FW → DEC when confidence above median
    'KO→DEC in MW/WW/FW (conf≥65)': (f) => {
        if (f.predMethod === 'KO' && ['MW', 'WW', 'FW'].includes(f.wc) && f.confidence >= 65) return 'DEC';
        return null;
    },

    // Rule 5: Non-volatile KO → DEC (non-HW)
    // Volatile fights are already getting DEC boost; non-volatile KOs are the ones over-predicted
    'Non-volatile KO→DEC (conf≥75, non-HW)': (f) => {
        // We approximate "non-volatile" as confidence >= 65 (above CLOSE_FIGHT_THRESHOLD)
        if (f.predMethod === 'KO' && f.confidence >= 75 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        return null;
    },

    // === COMBINED RULES ===
    // Best single (conf≥85) + BW/FLW catch-all
    'COMBO: conf≥85 non-HW + BW/FLW all KO': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 85 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        if (f.predMethod === 'KO' && ['BW', 'FLW'].includes(f.wc)) return 'DEC';
        return null;
    },

    // conf≥85 non-HW + MW/WW KO at high conf
    'COMBO: conf≥85 non-HW + MW/WW KO≥75': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 85 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        if (f.predMethod === 'KO' && ['MW', 'WW'].includes(f.wc) && f.confidence >= 75) return 'DEC';
        return null;
    },

    // Aggressive: conf≥80 non-HW + BW/FLW all KO
    'COMBO: conf≥80 non-HW + BW/FLW all KO': (f) => {
        if (f.predMethod === 'KO' && f.confidence >= 80 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
        if (f.predMethod === 'KO' && ['BW', 'FLW'].includes(f.wc)) return 'DEC';
        return null;
    },

    // Conservative: conf≥85 only (already best single) — included for comparison
    // (duplicate of above, just for labeling in combined section)
};

// === RUN BACKTESTS ===
console.log('='.repeat(90));
console.log('RULE'.padEnd(40), 'Fixes'.padEnd(8), 'Breaks'.padEnd(8), 'Net Δ'.padEnd(8), 'ΔScore'.padEnd(10), 'New M%');
console.log('='.repeat(90));

// Baseline
const baselineMethodCorrect = matched.filter(f => f.methodCorrect).length;
const baselineRoundCorrect = matched.filter(f => f.roundCorrect).length;
const baselineScore = matched.reduce((s, f) => s + calcScore(f.winnerCorrect, f.methodCorrect, f.roundCorrect), 0);
console.log('BASELINE'.padEnd(40), ''.padEnd(8), ''.padEnd(8), ''.padEnd(8),
    String(baselineScore).padEnd(10),
    (baselineMethodCorrect / matched.length * 100).toFixed(1) + '%');
console.log('-'.repeat(90));

const ruleResults = {};
for (const [ruleName, ruleFn] of Object.entries(rules)) {
    let fixes = 0, breaks = 0, scoreChange = 0;
    const affectedFights = [];

    for (const f of matched) {
        const newMethod = ruleFn(f);
        if (newMethod === null) continue; // Rule doesn't apply

        const oldMethodCorrect = f.methodCorrect;
        const newMethodCorrect = f.winnerCorrect && newMethod === f.actualMethod;
        const oldRoundCorrect = f.roundCorrect;
        // If method changes to DEC, round is auto-correct for DEC results
        const newRoundCorrect = newMethodCorrect && (newMethod === 'DEC' ? true : f.predRound === f.actualRound);

        const oldScore = calcScore(f.winnerCorrect, oldMethodCorrect, oldRoundCorrect);
        const newScore = calcScore(f.winnerCorrect, newMethodCorrect, newRoundCorrect);

        if (newMethodCorrect && !oldMethodCorrect) {
            fixes++;
            affectedFights.push({ ...f, delta: 'FIX', scoreDelta: newScore - oldScore });
        } else if (!newMethodCorrect && oldMethodCorrect) {
            breaks++;
            affectedFights.push({ ...f, delta: 'BREAK', scoreDelta: newScore - oldScore });
        }
        scoreChange += newScore - oldScore;
    }

    const newMethodCorrect = baselineMethodCorrect + fixes - breaks;
    console.log(
        ruleName.padEnd(40),
        String(fixes).padEnd(8),
        String(breaks).padEnd(8),
        (fixes - breaks >= 0 ? '+' : '') + String(fixes - breaks).padEnd(7),
        (scoreChange >= 0 ? '+' : '') + String(scoreChange).padEnd(9),
        (newMethodCorrect / matched.length * 100).toFixed(1) + '%'
    );

    ruleResults[ruleName] = { fixes, breaks, scoreChange, affectedFights, newMethodCorrect };
}

// === DETAIL VIEW: Show best rule's individual fight impacts ===
console.log('\n');

// Find best rule by net score change
const bestRule = Object.entries(ruleResults).sort((a, b) => b[1].scoreChange - a[1].scoreChange)[0];
console.log(`=== BEST RULE: "${bestRule[0]}" (net ${bestRule[1].scoreChange >= 0 ? '+' : ''}${bestRule[1].scoreChange} pts) ===\n`);

console.log('FIXES:');
bestRule[1].affectedFights.filter(f => f.delta === 'FIX').forEach(f => {
    console.log(`  ${f.wc.padEnd(4)} ${f.predWinner.split(' ').pop().padEnd(15)} ` +
        `pred:${f.predMethod} → actual:${f.actualMethod}${f.actualMethod !== 'DEC' ? ' R' + f.actualRound : ''} ` +
        `| conf:${f.confidence.toFixed(0)}% | +${f.scoreDelta}pts | ${f.event.substring(0, 35)}`);
});

console.log('\nBREAKS:');
const breaks = bestRule[1].affectedFights.filter(f => f.delta === 'BREAK');
if (breaks.length === 0) {
    console.log('  (none)');
} else {
    breaks.forEach(f => {
        console.log(`  ${f.wc.padEnd(4)} ${f.predWinner.split(' ').pop().padEnd(15)} ` +
            `pred:${f.predMethod} → actual:${f.actualMethod}${f.actualMethod !== 'DEC' ? ' R' + f.actualRound : ''} ` +
            `| conf:${f.confidence.toFixed(0)}% | ${f.scoreDelta}pts | ${f.event.substring(0, 35)}`);
    });
}

// === ADDITIONAL ANALYSIS: What about fights we predicted DEC but finished? ===
console.log('\n\n=== DEC PREDICTED → ACTUAL FINISH (winner correct) ===');
const decToFinish = matched.filter(f => f.winnerCorrect && f.predMethod === 'DEC' && f.actualMethod !== 'DEC');
console.log(`${decToFinish.length} fights where DEC was predicted but a finish happened (winner correct):`);
decToFinish.forEach(f => {
    console.log(`  ${f.wc.padEnd(4)} ${f.predWinner.split(' ').pop().padEnd(15)} ` +
        `actual:${f.actualMethod} R${f.actualRound} | conf:${f.confidence.toFixed(0)}% | ${f.event.substring(0, 35)}`);
});

// === SUMMARY STATS ===
console.log('\n\n=== CURRENT MODEL STATS ===');
console.log(`Winner: ${matched.filter(f => f.winnerCorrect).length}/${matched.length} (${(matched.filter(f => f.winnerCorrect).length / matched.length * 100).toFixed(1)}%)`);
console.log(`Method: ${baselineMethodCorrect}/${matched.length} (${(baselineMethodCorrect / matched.length * 100).toFixed(1)}%)`);
console.log(`Round:  ${baselineRoundCorrect}/${matched.length} (${(baselineRoundCorrect / matched.length * 100).toFixed(1)}%)`);
console.log(`Score:  ${baselineScore}`);

const predMethodDist = { DEC: 0, KO: 0, SUB: 0 };
const actMethodDist = { DEC: 0, KO: 0, SUB: 0 };
matched.forEach(f => {
    predMethodDist[f.predMethod] = (predMethodDist[f.predMethod] || 0) + 1;
    actMethodDist[f.actualMethod] = (actMethodDist[f.actualMethod] || 0) + 1;
});
console.log(`\nPredicted method dist: DEC=${predMethodDist.DEC} (${(predMethodDist.DEC/matched.length*100).toFixed(0)}%), KO=${predMethodDist.KO} (${(predMethodDist.KO/matched.length*100).toFixed(0)}%), SUB=${predMethodDist.SUB} (${(predMethodDist.SUB/matched.length*100).toFixed(0)}%)`);
console.log(`Actual method dist:    DEC=${actMethodDist.DEC} (${(actMethodDist.DEC/matched.length*100).toFixed(0)}%), KO=${actMethodDist.KO} (${(actMethodDist.KO/matched.length*100).toFixed(0)}%), SUB=${actMethodDist.SUB} (${(actMethodDist.SUB/matched.length*100).toFixed(0)}%)`);
console.log(`Gap: DEC ${predMethodDist.DEC - actMethodDist.DEC > 0 ? 'over' : 'under'}-predicted by ${Math.abs(predMethodDist.DEC - actMethodDist.DEC)}, KO ${predMethodDist.KO - actMethodDist.KO > 0 ? 'over' : 'under'}-predicted by ${Math.abs(predMethodDist.KO - actMethodDist.KO)}, SUB ${predMethodDist.SUB - actMethodDist.SUB > 0 ? 'over' : 'under'}-predicted by ${Math.abs(predMethodDist.SUB - actMethodDist.SUB)}`);

// === PER-EVENT BREAKDOWN: v12 rule (conf≥80 non-HW + BW/FLW) ===
console.log('\n\n' + '='.repeat(90));
console.log('=== v12 PER-EVENT BREAKDOWN ===');
console.log('='.repeat(90));

const v12Rule = (f) => {
    if (f.predMethod === 'KO' && f.confidence >= 80 && f.wc !== 'HW' && f.wc !== 'LHW') return 'DEC';
    if (f.predMethod === 'KO' && ['BW', 'FLW'].includes(f.wc)) return 'DEC';
    return null;
};

// Group fights by event
const eventOrder = data.events.map(e => e.name.replace('UFC ', ''));
const fightsByEvent = {};
for (const f of matched) {
    const evShort = f.event;
    if (!fightsByEvent[evShort]) fightsByEvent[evShort] = [];
    fightsByEvent[evShort].push(f);
}

let totalV12Score = 0;
let totalV12Method = 0;
let totalV12Round = 0;

for (const evName of eventOrder) {
    const evShort = evName.replace('UFC ', '');
    const evFights = fightsByEvent[evShort];
    if (!evFights) continue;

    let baseW = 0, baseM = 0, baseR = 0, baseS = 0;
    let v12W = 0, v12M = 0, v12R = 0, v12S = 0;
    const changes = [];

    for (const f of evFights) {
        const bScore = calcScore(f.winnerCorrect, f.methodCorrect, f.roundCorrect);
        baseW += f.winnerCorrect ? 1 : 0;
        baseM += f.methodCorrect ? 1 : 0;
        baseR += f.roundCorrect ? 1 : 0;
        baseS += bScore;

        const newMethod = v12Rule(f);
        let v12MethodCorrect, v12RoundCorrect;
        if (newMethod !== null) {
            v12MethodCorrect = f.winnerCorrect && newMethod === f.actualMethod;
            v12RoundCorrect = v12MethodCorrect && (newMethod === 'DEC' ? true : f.predRound === f.actualRound);
        } else {
            v12MethodCorrect = f.methodCorrect;
            v12RoundCorrect = f.roundCorrect;
        }
        const v12Score = calcScore(f.winnerCorrect, v12MethodCorrect, v12RoundCorrect);

        v12W += f.winnerCorrect ? 1 : 0;
        v12M += v12MethodCorrect ? 1 : 0;
        v12R += v12RoundCorrect ? 1 : 0;
        v12S += v12Score;

        if (newMethod !== null && v12MethodCorrect !== f.methodCorrect) {
            changes.push({
                fighter: f.predWinner.split(' ').pop(),
                wc: f.wc,
                from: f.predMethod,
                actual: f.actualMethod,
                delta: v12Score - bScore,
                type: v12MethodCorrect ? 'FIX' : 'BREAK'
            });
        }
    }

    totalV12Score += v12S;
    totalV12Method += v12M;
    totalV12Round += v12R;

    const n = evFights.length;
    const scoreDelta = v12S - baseS;
    const methodDelta = v12M - baseM;
    console.log(`\n${evShort}`);
    console.log(`  v11: W ${baseW}/${n} M ${baseM}/${n} R ${baseR}/${n} | Score: ${baseS}`);
    console.log(`  v12: W ${v12W}/${n} M ${v12M}/${n} R ${v12R}/${n} | Score: ${v12S} (${scoreDelta >= 0 ? '+' : ''}${scoreDelta})`);
    if (changes.length > 0) {
        changes.forEach(c => {
            console.log(`    ${c.type === 'FIX' ? '+' : '-'} ${c.wc} ${c.fighter.padEnd(15)} ${c.from}→DEC (actual: ${c.actual}) ${c.delta >= 0 ? '+' : ''}${c.delta}pts`);
        });
    }
}

console.log('\n' + '='.repeat(90));
console.log(`TOTAL v11: M ${baselineMethodCorrect}/${matched.length} (${(baselineMethodCorrect/matched.length*100).toFixed(1)}%) R ${baselineRoundCorrect}/${matched.length} (${(baselineRoundCorrect/matched.length*100).toFixed(1)}%) | Score: ${baselineScore}`);
console.log(`TOTAL v12: M ${totalV12Method}/${matched.length} (${(totalV12Method/matched.length*100).toFixed(1)}%) R ${totalV12Round}/${matched.length} (${(totalV12Round/matched.length*100).toFixed(1)}%) | Score: ${totalV12Score} (${totalV12Score - baselineScore >= 0 ? '+' : ''}${totalV12Score - baselineScore})`);
console.log('='.repeat(90));

// v12 method distribution
const v12MethodDist = { DEC: 0, KO: 0, SUB: 0 };
matched.forEach(f => {
    const newM = v12Rule(f);
    v12MethodDist[newM || f.predMethod]++;
});
console.log(`\nv12 predicted dist: DEC=${v12MethodDist.DEC} (${(v12MethodDist.DEC/matched.length*100).toFixed(0)}%), KO=${v12MethodDist.KO} (${(v12MethodDist.KO/matched.length*100).toFixed(0)}%), SUB=${v12MethodDist.SUB} (${(v12MethodDist.SUB/matched.length*100).toFixed(0)}%)`);
console.log(`Actual dist:        DEC=${actMethodDist.DEC} (${(actMethodDist.DEC/matched.length*100).toFixed(0)}%), KO=${actMethodDist.KO} (${(actMethodDist.KO/matched.length*100).toFixed(0)}%), SUB=${actMethodDist.SUB} (${(actMethodDist.SUB/matched.length*100).toFixed(0)}%)`);
console.log(`v12 gap: DEC ${v12MethodDist.DEC - actMethodDist.DEC > 0 ? 'over' : 'under'}-predicted by ${Math.abs(v12MethodDist.DEC - actMethodDist.DEC)}, KO ${v12MethodDist.KO - actMethodDist.KO > 0 ? 'over' : 'under'}-predicted by ${Math.abs(v12MethodDist.KO - actMethodDist.KO)}`);
