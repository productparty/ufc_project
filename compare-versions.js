const fs = require('fs');
const dump = JSON.parse(fs.readFileSync('results/db-dump.json','utf8'));
const engineCode = fs.readFileSync('prediction-engine.js', 'utf8');
const PredictionEngine = new Function(engineCode + '\nreturn PredictionEngine;')();
const engine = new PredictionEngine();

const stricklandId = dump.events.find(e => e.name.includes('Strickland')).id;
const fights = dump.fights.filter(f => f.eventId === stricklandId);
const v4Preds = engine.generatePredictions(fights, 'fight-night');

// Old predictions from DB (v2)
const oldPreds = dump.predictions.filter(p => p.eventId === stricklandId);

console.log('Fight'.padEnd(40), 'WC'.padEnd(5), 'v2 (stored)'.padEnd(18), 'v4 (current)'.padEnd(18), 'Change?');
console.log('-'.repeat(100));

let v2Finish = 0, v4Finish = 0;
fights.forEach((f, i) => {
    const v4 = v4Preds[i];
    const old = oldPreds.find(p => p.fightId === f.id);
    if (!old) return;

    const oldStr = old.method + (old.round !== 'DEC' ? ' ' + old.round : '');
    const v4Str = v4.method + (v4.round !== 'DEC' ? ' ' + v4.round : '');

    if (old.method !== 'DEC') v2Finish++;
    if (v4.method !== 'DEC') v4Finish++;

    const changed = (old.method !== v4.method || old.round !== v4.round) ? '<<<' : '';
    const fightName = f.fighterA.name.split(' ').pop() + ' vs ' + f.fighterB.name.split(' ').pop();
    console.log(fightName.padEnd(40), f.weightClass.padEnd(5), oldStr.padEnd(18), v4Str.padEnd(18), changed);
});

console.log();
console.log('v2 finishes:', v2Finish + '/14 (' + (v2Finish/14*100).toFixed(0) + '%)');
console.log('v4 finishes:', v4Finish + '/14 (' + (v4Finish/14*100).toFixed(0) + '%)');

// Show reasoning for any fight that changed
console.log('\n--- Reasoning for Changed Predictions ---');
fights.forEach((f, i) => {
    const v4 = v4Preds[i];
    const old = oldPreds.find(p => p.fightId === f.id);
    if (!old) return;
    if (old.method === v4.method && old.round === v4.round) return;

    const fightName = f.fighterA.name + ' vs ' + f.fighterB.name;
    console.log('\n' + fightName + ' (' + f.weightClass + ')');
    console.log('  v2:', old.winnerName, old.method, old.round);
    console.log('  v4:', v4.winnerName, v4.method, v4.round, '(conf: ' + v4.confidence.toFixed(1) + '%)');

    // Print key reasoning
    if (v4.reasoning) {
        const allReasons = [...(v4.reasoning.method || []), ...(v4.reasoning.round || [])];
        allReasons.forEach(r => {
            if (r.text) console.log('    ' + r.text);
        });
    }
});
