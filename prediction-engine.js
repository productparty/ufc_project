/**
 * UFC Weekly Predictor - 3-Layer Prediction Engine
 * Implements the complete UFC Fight Prediction Model specification
 */

class PredictionEngine {
    constructor() {
        // Model version — bump this when tuning parameters
        // v3: scoring, age wall, year-off, SApM, TD defense, source reweighting
        // v4: close fight threshold, base rate prior, method blend, small sample regression
        // v5: DEC EV multiplier (1.30)
        // v6: R1 thresholds lowered 4-5pts, volatile DEC boost (1.10)
        // v7: Striking efficiency modifier in Layer 1 (net striking differential + TD defense amplifier)
        // v8: Scoring-optimized DEC multiplier (1.30→1.25) based on league scoring reverse-engineering
        //     League scoring: W=150, M=125 (gated by W), R=50 (gated by W)
        //     DEC correct = 175 EV (125 method + 50 round guaranteed)
        //     Finish correct = 140 EV (125 method + ~15 round at 30% accuracy)
        //     True DEC premium = 175/140 = 1.25
        // v9: Division-aware tuning — division-specific source weights, confidence calibration,
        //     and close-fight DEC caps to address LW (37.5%) and HW (50%) underperformance
        // v10: Event-structure DEC adjustments based on UFC 326 top-scorer analysis
        //     5-round DEC modifier 1.10→1.15, PPV main event 1.15→1.20, volatile boost 1.10→1.12
        // v11: Method accuracy tuning based on league standings analysis
        //     Volatile DEC boost 1.12→1.05 (v10 over-predicted DEC: 76.9% vs 69.2% actual)
        //     highConfidenceDisagreement threshold 15→25 (weak dissent shouldn't force DEC)
        //     Volatility trigger tightened: require 2+ dissenting sources OR 1 strong (>60%)
        //     SUB specialist boost when Tapology SUB ≥55% + betting fav ≥65%
        // v12: High-confidence DEC regression + BW/FLW finish penalty
        //     103-fight backtest: model KO 38% vs actual 28% (10pt over-prediction)
        //     Compounding finish multipliers inflate KO past DEC EV premium
        //     Key insight: high winner confidence predicts DEC not finishes
        //     HC regression: conf≥80 non-HW/LHW → 1.30x additional DEC boost
        //     BW/FLW blanket finish penalty 0.75x (KO accuracy ~20-25% in these divisions)
        //     Backtest: +5 net method fixes, +975 pts over 103 fights
        // v13: R1 threshold reduction — partially reverting v11's 1.2x R1 scaling to ~1.1x
        //     9 events (116 fights): R1 is 54.4% of 3-round finishes, but model consistently
        //     predicts R2 when R1 occurs (McKinney R2→R1, Chiesa R2→R1, etc.)
        //     3-round R1 thresholds lowered by 4pts, 5-round by 5pts across all divisions
        //     R2+ thresholds unchanged — only affects R1 accessibility
        //     KO/DEC gap narrowed from 10pts to ~4pts (v12 working, no DEC EV changes)
        // v14: 10-event analysis (129 fights) — SUB under-prediction fix + DEC gravity reduction
        //     SUB predicted 8% vs actual 12.4% (biggest persistent gap)
        //     DEC over-prediction varies ±35pt per event from compounding multipliers
        //     67% of fights flagged volatile — threshold too loose, feeds DEC gravity
        //     Key changes:
        //     - Volatile DEC boost removed (1.05→1.0) — was permanent DEC tax on 67% of fights
        //     - Close-fight DEC caps reduced ~15% across all divisions
        //     - BW/FLW finish penalty softened 0.75→0.85 (BW has 85.7% winner accuracy)
        //     - SUB specialist threshold lowered 55→40% with tiered boosts
        //     - HC DEC regression 1.30→1.10, now mutually exclusive with lopsided finish boost
        //     - LW confidence discount 0.80→0.75 (35% winner accuracy)
        //     - Close-margin volatility tightened (2→3 sources required)
        //     - SUB base rate raised 20→23% in method prior
        //     - BestFightOdds replaces DRatings as data source (method props + moneyline)
        this.MODEL_VERSION = 'v14';

        // Confidence thresholds
        this.CONFIDENCE_HIGH = 65;
        this.CONFIDENCE_MEDIUM = 55;

        // Finish thresholds
        this.FINISH_THRESHOLD = 65; // % career finishes needed for finish prediction
        this.OPPONENT_FINISH_LOSS_THRESHOLD = 50; // % finish losses for opponent

        // Source agreement thresholds
        this.STRONG_AGREEMENT_THRESHOLD = 4; // 4+ sources agree on same winner
        this.LOPSIDED_FAVORITE_THRESHOLD = 80; // confidence above this = likely finish
        this.CLOSE_FIGHT_THRESHOLD = 63; // confidence below this = lean DEC (raised from 58 in v4)

        // Layoff thresholds (strengthened)
        this.LAYOFF_MODERATE = 300; // days - small penalty
        this.LAYOFF_SEVERE = 400; // days - moderate penalty
        this.LAYOFF_EXTREME = 500; // days - heavy penalty

        // v8: Scoring-optimized DEC certainty premium
        // League scoring: W=150, M=125 (gated by W), R=50 (gated by W)
        // DEC correct EV = 175 (method 125 + round 50 auto-correct)
        // Finish correct EV = 140 (method 125 + round 50 × ~30% accuracy)
        // DEC premium = 175/140 = 1.25 (was 1.30 — over-favored DEC, missed callable finishes)
        this.DEC_EV_MULTIPLIER = 1.25;

        // v12: High-confidence DEC regression
        // Backtest over 103 fights: model predicts KO 38% vs actual 28%. Compounding finish
        // multipliers inflate KO probability past the DEC EV multiplier. High winner confidence
        // actually predicts DEC, not finishes — dominant favorites cruise rather than chase KOs.
        // Regression applies an additional DEC multiplier when confidence is high in non-HW/LHW.
        this.HC_DEC_REGRESSION_THRESHOLD = 80;  // confidence % above which regression applies
        this.HC_DEC_REGRESSION_MULTIPLIER = 1.10; // v14: 1.30→1.10, now mutually exclusive with lopsided finish boost

        // v12: Division-specific KO suppression for decision-heavy divisions
        // BW/FLW KO predictions are correct only 20-25% of the time (most go DEC).
        // Apply a blanket finish penalty in these divisions.
        this.DIVISION_FINISH_PENALTY = {
            'BW':   0.85,   // v14: 0.75→0.85 — BW has 85.7% winner accuracy, finishes were over-suppressed
            'FLW':  0.85,   // v14: 0.75→0.85 — same adjustment
        };

        // v9: Division-specific close-fight DEC boost caps
        // Close fights lean toward DEC, but HOW MUCH depends on the division's finish rate
        // LW/FW/BW: decision-heavy → aggressive DEC boost in close fights
        // HW/LHW: finish-heavy → mild DEC boost even in close fights
        // v14: Caps reduced ~15% for men's, ~10% for women's — compounding DEC multipliers
        // caused ±35pt per-event variance. Close fights still lean DEC but less aggressively.
        this.DIVISION_CLOSE_FIGHT_DEC_CAP = {
            'HW':   1.20,   // was 1.30 — 70% finish rate
            'LHW':  1.30,   // was 1.40 — 62% finish rate
            'MW':   1.35,   // was 1.50 — 59% finish rate
            'WW':   1.35,   // was 1.50 — 52% finish rate
            'LW':   1.55,   // was 1.70 — moderated (compounds with LW conf discount)
            'FW':   1.45,   // was 1.65 — 46% finish rate
            'BW':   1.45,   // was 1.60 — 45% finish rate
            'FLW':  1.45,   // was 1.60 — 46% finish rate
            'WSW':  1.65,   // was 1.80 — 33% finish rate
            'WFLW': 1.60,   // was 1.75 — 36% finish rate
            'WBW':  1.60,   // was 1.75 — 39% finish rate
            'WFW':  1.50    // was 1.65 — ~45% finish rate
        };

        // v9: Division-specific source weight overrides
        // Different rating systems have different predictive power per division
        // Based on analysis: betting odds strong in HW (market prices KO risk well),
        // Tapology stronger in lighter divisions (community reads skill matchups better)
        this.DIVISION_SOURCE_WEIGHTS = {
            'HW':  { tapology: 0.10, bfo: 0.10, dratings: 0.10, betting: 0.26, eloK170: 0.20, eloMod: 0.13, glicko: 0.12, whr: 0.09 },
            'LHW': { tapology: 0.11, bfo: 0.11, dratings: 0.11, betting: 0.24, eloK170: 0.19, eloMod: 0.12, glicko: 0.12, whr: 0.11 },
            'LW':  { tapology: 0.18, bfo: 0.14, dratings: 0.14, betting: 0.16, eloK170: 0.14, eloMod: 0.12, glicko: 0.14, whr: 0.12 },
            'FW':  { tapology: 0.16, bfo: 0.13, dratings: 0.13, betting: 0.18, eloK170: 0.16, eloMod: 0.12, glicko: 0.13, whr: 0.12 },
        };

        // v9: Division-confidence calibration - discount confidence for historically weak divisions
        // Shrinks composite confidence toward 50% (less decisive picks) for divisions we do poorly in
        // Factor < 1.0 = pull toward 50% (weaker divisions); 1.0 = no change
        this.DIVISION_CONFIDENCE_CALIBRATION = {
            'BW':   1.00,   // 93.3% accuracy - trust fully
            'MW':   1.00,   // 81.8% accuracy - trust fully
            'FLW':  1.00,   // 75.0% accuracy - trust fully
            'FW':   0.95,   // 66.7% accuracy - slight caution
            'WW':   0.95,   // 66.7% accuracy - slight caution
            'HW':   0.85,   // 50.0% accuracy - meaningful discount
            'LW':   0.75,   // v14: 0.80→0.75 — 35% accuracy warrants stronger discount
            'LHW':  0.90,   // limited data, mild caution
            'WSW':  0.85,   // limited data, women's = more upsets
            'WFLW': 0.85,
            'WBW':  0.85,
            'WFW':  0.90
        };

        // Method base rate prior - historical UFC averages (~50% DEC, 30% KO, 20% SUB)
        // Acts as Bayesian anchor to prevent finish over-prediction
        this.METHOD_BASE_RATE = { ko: 33, sub: 23, dec: 44 }; // v14: SUB 20→23, DEC 47→44 — SUB under-predicted by 4.6pts
        this.METHOD_PRIOR_WEIGHT = 0.30;

        // Weight class finish bias - calibrated from historical UFC rates
        // Historical finish%: HW 70%, LHW 62%, MW 59%, WW 52%, LW 51%, FW 46%, BW 45%, FLW 46%
        // Female: WBW 39%, WFLW 36%, WSW 33%
        this.WEIGHT_CLASS_FINISH_BIAS = {
            'HW': { ko: 1.35, sub: 1.10, dec: 0.70 },   // 70% finish rate, 48% TKO
            'LHW': { ko: 1.20, sub: 1.00, dec: 0.85 },   // 62% finish rate, 43% TKO
            'MW': { ko: 1.05, sub: 1.05, dec: 0.92 },    // 59% finish rate, 37% TKO
            'WW': { ko: 1.00, sub: 1.00, dec: 1.00 },    // 52% finish rate (baseline)
            'LW': { ko: 0.95, sub: 1.05, dec: 1.02 },    // 51% finish rate, 29% TKO, 22% SUB
            'FW': { ko: 0.92, sub: 0.95, dec: 1.10 },    // 46% finish rate, 28% TKO
            'BW': { ko: 0.88, sub: 0.97, dec: 1.12 },    // 45% finish rate, 26% TKO
            'FLW': { ko: 0.87, sub: 1.05, dec: 1.08 },   // 46% finish rate, 25% TKO, 22% SUB
            'WSW': { ko: 0.70, sub: 0.85, dec: 1.35 },   // 33% finish rate, 13% TKO — female penalty
            'WFLW': { ko: 0.72, sub: 0.88, dec: 1.30 },  // 36% finish rate, 17% TKO — female penalty
            'WBW': { ko: 0.75, sub: 0.82, dec: 1.28 },   // 39% finish rate, 23% TKO — female penalty
            'WFW': { ko: 0.80, sub: 0.90, dec: 1.20 }    // ~45% finish rate (limited data)
        };

        // Round prediction: continuous scoring constants (aligned with Python implementation)
        // v11: 0.50→0.60 — v11 finish specialist boosts let moderate finishes through (50-55% KO),
        // but at 0.50 scale those all collapsed into R3. At 0.60, strong finishes can reach R2
        // while moderate ones still default to R3. R1 remains reserved for overwhelming signals.
        this.METHOD_CONFIDENCE_SCALE = 0.60;
        this.BONUS_LOSER_POWER_PUNCHER = 5.0;
        this.BONUS_LOPSIDED = 5.0;
        this.MAX_BONUS_CAP = 8.0;

        // Division-specific round thresholds for 3-round fights
        // Calibrated from historical UFC data:
        //   Overall 3-round finish distribution: R1 54.4%, R2 30.7%, R3 14.9%
        //   Higher threshold = harder to predict that round
        //   HW: 47% of KOs in R1, keep R1 accessible but not too easy
        //   BW/WBW: rare early finishes, 96% WBW fights go Over 1.5 rounds
        //   WW/FLW: front-loaded finishes, lower R1 thresholds
        // v6: R1 thresholds lowered by 4-5pts — model was over-predicting R2 when R1 finishes occurred
        //   (Gandra R2→R1, Pinas R2→R1, multiple across earlier events)
        // v11: R1 thresholds scaled ×1.2 (conservative — R1 is high-risk).
        // v13: R1 thresholds reduced by 4pts (3-round) / 5pts (5-round) — partial revert of v11.
        //   v11's 1.2x was too conservative; R1 is 54.4% of 3-round finishes but under-predicted.
        // R2 thresholds kept near original (×1.0) so strong finishers can reach R2.
        // Net effect: R2 is now reachable for 60%+ KO/SUB with bonuses, R3 remains default for moderate.
        this.DIVISION_ROUND_THRESHOLDS_3RD = {
            'HW':   { R1: 39.0, R2: 31.0, R3: 0.0 },   // 47% of HW KOs in R1
            'LHW':  { R1: 43.0, R2: 34.0, R3: 0.0 },   // High finish rate
            'MW':   { R1: 49.0, R2: 39.0, R3: 0.0 },   // Moderate
            'WW':   { R1: 46.0, R2: 37.0, R3: 0.0 },   // R1-heavy finishes
            'LW':   { R1: 50.0, R2: 41.0, R3: 0.0 },   // Moderate
            'FW':   { R1: 52.0, R2: 43.0, R3: 0.0 },   // Below average finish rate
            'BW':   { R1: 58.0, R2: 45.0, R3: 0.0 },   // Low early finish rate
            'FLW':  { R1: 49.0, R2: 39.0, R3: 0.0 },   // Front-loaded when finishes occur
            'WSW':  { R1: 63.0, R2: 52.0, R3: 0.0 },   // Very low finish rate + female
            'WFLW': { R1: 61.0, R2: 50.0, R3: 0.0 },   // Low finish rate + female
            'WBW':  { R1: 66.0, R2: 52.0, R3: 0.0 },   // 96% go Over 1.5 rounds
            'WFW':  { R1: 58.0, R2: 47.0, R3: 0.0 }    // Limited data
        };

        // Division-specific round thresholds for 5-round fights
        // 5-round distribution is much flatter: R1 31.1%, R2 26.7%, R3 22.2%, R4 8.9%, R5 11.1%
        // All thresholds raised significantly vs 3-round — fighters pace for 25min
        // v6: R1 thresholds lowered by 4pts to reduce R2→R1 mispredictions in title fights
        // v11: R1 scaled ×1.2, R2/R3 kept near original to open up earlier rounds
        this.DIVISION_ROUND_THRESHOLDS_5RD = {
            'HW':   { R1: 48.0, R2: 39.0, R3: 29.0, R4: 18.0, R5: 0.0 },
            'LHW':  { R1: 50.0, R2: 41.0, R3: 31.0, R4: 18.0, R5: 0.0 },
            'MW':   { R1: 55.0, R2: 45.0, R3: 35.0, R4: 22.0, R5: 0.0 },
            'WW':   { R1: 53.0, R2: 43.0, R3: 33.0, R4: 19.0, R5: 0.0 },
            'LW':   { R1: 57.0, R2: 47.0, R3: 36.0, R4: 22.0, R5: 0.0 },
            'FW':   { R1: 57.0, R2: 47.0, R3: 36.0, R4: 22.0, R5: 0.0 },
            'BW':   { R1: 62.0, R2: 49.0, R3: 39.0, R4: 24.0, R5: 0.0 },
            'FLW':  { R1: 60.0, R2: 47.0, R3: 36.0, R4: 22.0, R5: 0.0 },
            'WSW':  { R1: 67.0, R2: 54.0, R3: 42.0, R4: 26.0, R5: 0.0 },
            'WFLW': { R1: 65.0, R2: 52.0, R3: 40.0, R4: 26.0, R5: 0.0 },
            'WBW':  { R1: 67.0, R2: 54.0, R3: 42.0, R4: 26.0, R5: 0.0 },
            'WFW':  { R1: 62.0, R2: 49.0, R3: 39.0, R4: 24.0, R5: 0.0 }
        };

        this.FALLBACK_THRESHOLDS_3RD = { R1: 55.0, R2: 41.0, R3: 0.0 };
        this.FALLBACK_THRESHOLDS_5RD = { R1: 61.0, R2: 46.0, R3: 35.0, R4: 22.0, R5: 0.0 };

        // Grappler detection thresholds
        this.WRESTLER_TD_THRESHOLD = 2.5; // TDs per 15 min
        this.WRESTLER_SUB_WIN_THRESHOLD = 50; // % sub wins
        this.VETERAN_CONTROL_TD_THRESHOLD = 2.0;
        this.VETERAN_CONTROL_TIME_THRESHOLD = 2.0; // mins per round

        // Early KO threat multiplier conditions
        this.EARLY_KO_THREAT_TAPOLOGY_THRESHOLD = 65; // underdog < this
        this.EARLY_KO_KO_WIN_THRESHOLD = 75; // % KO wins needed
    }

    /**
     * Generate predictions for all fights in an event
     */
    generatePredictions(fights, eventType) {
        return fights.map(fight => this.predictFight(fight, eventType));
    }

    /**
     * Generate prediction for a single fight
     */
    predictFight(fight, eventType) {
        // Infer missing data from opponent before predicting
        this.inferMissingFromOpponent(fight);

        const reasoning = [];

        // Layer 1: Winner Selection
        const layer1Result = this.layer1WinnerSelection(fight, reasoning);

        // Layer 2: Method Selection
        const layer2Result = this.layer2MethodSelection(fight, layer1Result, eventType, reasoning);

        // Layer 3: Round Prediction
        const layer3Result = this.layer3RoundPrediction(fight, layer1Result, layer2Result, reasoning);

        // Determine which data sources contributed to this prediction
        const dataSources = this.getContributingSources(fight);

        return {
            fightId: fight.id,
            winner: layer1Result.winner,
            winnerName: layer1Result.winnerName,
            method: layer2Result.method,
            round: layer3Result.round,
            confidence: layer1Result.confidence,
            confidenceTier: layer1Result.confidenceTier,
            isVolatile: layer1Result.isVolatile,
            primarySource: layer1Result.primarySource,
            dataSources: dataSources,
            modelVersion: this.MODEL_VERSION,
            reasoning: {
                winner: reasoning.filter(r => r.layer === 1),
                method: reasoning.filter(r => r.layer === 2),
                round: reasoning.filter(r => r.layer === 3)
            }
        };
    }

    /**
     * Get list of sources that contributed data to this prediction
     */
    getContributingSources(fight) {
        const sources = [];
        const fighterA = fight.fighterA || {};
        const fighterB = fight.fighterB || {};

        // Tapology - check if consensus exists and isn't default 50
        if (fighterA.tapology?.consensus && fighterA.tapology.consensus !== 50) {
            sources.push('tapology');
        }

        // BFO - check if winPct exists
        if (fighterA.bfo?.winPct) {
            sources.push('bfo');
        }

        // DRatings - check if winPct exists and isn't default 50 (backward compat)
        const dratingsA = this.extractDRatingsWinPct(fighterA.dratings);
        if (dratingsA !== 50 && !fighterA.bfo?.winPct) {
            sources.push('dratings');
        }

        // FightMatrix expanded data
        if (fighterA.fightmatrix?.eloK170 || fighterB.fightmatrix?.eloK170) {
            sources.push('fightmatrix-elo');
        }
        if (fighterA.fightmatrix?.bettingWinPct || fighterB.fightmatrix?.bettingWinPct) {
            sources.push('betting-odds');
        }
        if (fighterA.fightmatrix?.age || fighterB.fightmatrix?.age) {
            sources.push('age-data');
        }

        // Legacy FightMatrix - check if CIRRS exists
        if (!sources.includes('fightmatrix-elo') && (fighterA.fightMatrix?.cirrs || fighterA.cirrs || fighterB.fightMatrix?.cirrs || fighterB.cirrs)) {
            sources.push('fightmatrix');
        }

        // UFCStats - check if any meaningful stats exist
        if (fighterA.ufcStats?.slpm || fighterA.ufcStats?.koWinPct !== null ||
            fighterB.ufcStats?.slpm || fighterB.ufcStats?.koWinPct !== null) {
            sources.push('ufcstats');
        }

        return sources;
    }

    /**
     * Layer 1: Winner Selection
     * Uses source agreement logic with Tapology override rule
     */
    layer1WinnerSelection(fight, reasoning) {
        const fighterA = fight.fighterA;
        const fighterB = fight.fighterB;

        // Calculate composite win probability
        const sources = this.gatherSourceData(fight);
        const composite = this.calculateCompositeWinProb(sources, fight.weightClass);

        // Calculate source agreement (new)
        const sourceAgreement = this.calculateSourceAgreement(sources);

        reasoning.push({
            layer: 1,
            type: 'source_data',
            text: `Sources - Tapology: ${fighterA.name} ${sources.tapologyA}% / ${fighterB.name} ${sources.tapologyB}%, DRatings: ${sources.dratingsA}% / ${sources.dratingsB}%`
        });

        // Log source agreement
        if (sourceAgreement.totalSources > 0) {
            reasoning.push({
                layer: 1,
                type: 'source_agreement',
                text: `Source Agreement: ${sourceAgreement.agreementCount}/${sourceAgreement.totalSources} sources agree${sourceAgreement.disagreingSources.length > 0 ? ` (dissenting: ${sourceAgreement.disagreingSources.join(', ')})` : ' (unanimous)'}`
            });
        }

        // Log FightMatrix rating systems
        if (sources.eloK170A && sources.eloK170B) {
            reasoning.push({
                layer: 1,
                type: 'fight_matrix',
                text: `FightMatrix Elo K170: ${fighterA.name} ${sources.eloK170A.winPct.toFixed(1)}% / ${fighterB.name} ${sources.eloK170B.winPct.toFixed(1)}%`
            });
        }

        if (sources.bettingWinPctA && sources.bettingWinPctB) {
            reasoning.push({
                layer: 1,
                type: 'betting_odds',
                text: `Betting Odds: ${fighterA.name} ${sources.bettingWinPctA.toFixed(1)}% / ${fighterB.name} ${sources.bettingWinPctB.toFixed(1)}%`
            });
        }

        // Log age and activity modifiers if applicable
        if (sources.ageA && sources.ageB) {
            reasoning.push({
                layer: 1,
                type: 'age_data',
                text: `Ages: ${fighterA.name} ${sources.ageA} / ${fighterB.name} ${sources.ageB}`
            });
        }

        if (sources.daysSinceLastFightA && sources.daysSinceLastFightB) {
            reasoning.push({
                layer: 1,
                type: 'activity_data',
                text: `Days since last fight: ${fighterA.name} ${sources.daysSinceLastFightA} / ${fighterB.name} ${sources.daysSinceLastFightB}`
            });
        }

        if (sources.last3RecordA && sources.last3RecordB) {
            reasoning.push({
                layer: 1,
                type: 'form_data',
                text: `Last 3 fights: ${fighterA.name} (${sources.last3RecordA}) / ${fighterB.name} (${sources.last3RecordB})`
            });
        }

        // UFCStats career data
        if (fighterA.ufcStats?.koWinPct !== null || fighterB.ufcStats?.koWinPct !== null) {
            const aKO = fighterA.ufcStats?.koWinPct?.toFixed(0) || 'N/A';
            const aSUB = fighterA.ufcStats?.subWinPct?.toFixed(0) || 'N/A';
            const aFinish = fighterA.ufcStats?.finishWinPct?.toFixed(0) || 'N/A';
            const bKO = fighterB.ufcStats?.koWinPct?.toFixed(0) || 'N/A';
            const bSUB = fighterB.ufcStats?.subWinPct?.toFixed(0) || 'N/A';
            const bFinish = fighterB.ufcStats?.finishWinPct?.toFixed(0) || 'N/A';
            reasoning.push({
                layer: 1,
                type: 'ufcstats_data',
                text: `UFCStats Career: ${fighterA.name} (KO ${aKO}%, SUB ${aSUB}%, Finish ${aFinish}%) / ${fighterB.name} (KO ${bKO}%, SUB ${bSUB}%, Finish ${bFinish}%)`
            });
        }

        if (fighterA.ufcStats?.slpm !== null || fighterB.ufcStats?.slpm !== null) {
            const aSlpm = fighterA.ufcStats?.slpm?.toFixed(2) || 'N/A';
            const aTdAvg = fighterA.ufcStats?.tdAvg?.toFixed(1) || 'N/A';
            const bSlpm = fighterB.ufcStats?.slpm?.toFixed(2) || 'N/A';
            const bTdAvg = fighterB.ufcStats?.tdAvg?.toFixed(1) || 'N/A';
            reasoning.push({
                layer: 1,
                type: 'ufcstats_activity',
                text: `UFCStats Activity: ${fighterA.name} (SLpM ${aSlpm}, TD ${aTdAvg}/15min) / ${fighterB.name} (SLpM ${bSlpm}, TD ${bTdAvg}/15min)`
            });
        }

        if (fighterA.ufcStats?.finishLossPct !== null || fighterB.ufcStats?.finishLossPct !== null) {
            const aFinishLoss = fighterA.ufcStats?.finishLossPct?.toFixed(0) || 'N/A';
            const bFinishLoss = fighterB.ufcStats?.finishLossPct?.toFixed(0) || 'N/A';
            reasoning.push({
                layer: 1,
                type: 'ufcstats_vulnerability',
                text: `UFCStats Vulnerability: ${fighterA.name} (${aFinishLoss}% finish losses) / ${fighterB.name} (${bFinishLoss}% finish losses)`
            });
        }

        // Legacy CIRRS fallback
        if (sources.fightMatrixA && sources.fightMatrixB && !sources.eloK170A) {
            const fmGap = sources.fightMatrixA - sources.fightMatrixB;
            reasoning.push({
                layer: 1,
                type: 'fight_matrix',
                text: `Fight Matrix CIRRS: ${fighterA.name} ${sources.fightMatrixA} / ${fighterB.name} ${sources.fightMatrixB} (Gap: ${fmGap > 0 ? '+' : ''}${fmGap})`
            });
        }

        // Log non-zero modifiers
        const mods = composite.modifiers;
        const activeModParts = [];
        if (mods.age !== 0) activeModParts.push(`Age ${mods.age > 0 ? '+' : ''}${mods.age.toFixed(1)}`);
        if (mods.activity !== 0) activeModParts.push(`Activity ${mods.activity > 0 ? '+' : ''}${mods.activity.toFixed(1)}`);
        if (mods.form !== 0) activeModParts.push(`Form ${mods.form > 0 ? '+' : ''}${mods.form.toFixed(1)}`);
        if (mods.striking !== 0) activeModParts.push(`Striking ${mods.striking > 0 ? '+' : ''}${mods.striking.toFixed(1)}`);
        if (activeModParts.length > 0) {
            reasoning.push({
                layer: 1,
                type: 'modifiers',
                text: `Modifiers (favor ${fighterA.name}): ${activeModParts.join(', ')}`
            });
        }

        // Determine winner based on composite
        let winner, winnerName, confidence, primarySource;
        let isVolatile = false;

        if (composite.winProbA >= 50) {
            winner = 'fighterA';
            winnerName = fighterA.name;
            confidence = composite.winProbA;
            primarySource = composite.primarySourceA;
        } else {
            winner = 'fighterB';
            winnerName = fighterB.name;
            confidence = composite.winProbB;
            primarySource = composite.primarySourceB;
        }

        // Tapology Override Rule: >75% consensus can override in close fights
        const tapologyOverrideThreshold = 75;
        const closeFightThreshold = 60;

        if (confidence < closeFightThreshold) {
            if (sources.tapologyA > tapologyOverrideThreshold && winner !== 'fighterA') {
                winner = 'fighterA';
                winnerName = fighterA.name;
                confidence = sources.tapologyA;
                primarySource = 'tapology';
                reasoning.push({
                    layer: 1,
                    type: 'override',
                    text: `Tapology Override: ${fighterA.name} has ${sources.tapologyA}% consensus (>${tapologyOverrideThreshold}%) overriding close fight (<${closeFightThreshold}%)`
                });
            } else if (sources.tapologyB > tapologyOverrideThreshold && winner !== 'fighterB') {
                winner = 'fighterB';
                winnerName = fighterB.name;
                confidence = sources.tapologyB;
                primarySource = 'tapology';
                reasoning.push({
                    layer: 1,
                    type: 'override',
                    text: `Tapology Override: ${fighterB.name} has ${sources.tapologyB}% consensus (>${tapologyOverrideThreshold}%) overriding close fight (<${closeFightThreshold}%)`
                });
            }
        }

        // Volatility Detection: sources disagree on winner
        const sourceDisagreement = this.checkSourceDisagreement(sources);
        if (sourceDisagreement) {
            isVolatile = true;
            reasoning.push({
                layer: 1,
                type: 'volatility',
                text: `Volatility Flag: Sources disagree - ${sourceDisagreement}`
            });
        }

        // Confidence tier
        let confidenceTier;
        if (confidence >= this.CONFIDENCE_HIGH) {
            confidenceTier = 'high';
        } else if (confidence >= this.CONFIDENCE_MEDIUM) {
            confidenceTier = 'medium';
        } else {
            confidenceTier = 'low';
            isVolatile = true;
        }

        reasoning.push({
            layer: 1,
            type: 'result',
            text: `Winner: ${winnerName} (${confidence.toFixed(1)}% confidence, ${confidenceTier} tier, primary source: ${primarySource})`
        });

        return {
            winner,
            winnerName,
            confidence,
            confidenceTier,
            isVolatile,
            primarySource,
            sourceAgreement // Include for Layer 2 and 3 to use
        };
    }

    /**
     * Layer 2: Method Selection
     * Applies finish thresholding rule, weight class bias, and source agreement modifiers
     */
    layer2MethodSelection(fight, layer1Result, eventType, reasoning) {
        const winner = layer1Result.winner;
        const winnerData = fight[winner] || {};
        const loserKey = winner === 'fighterA' ? 'fighterB' : 'fighterA';
        const loserData = fight[loserKey] || {};
        const confidence = layer1Result.confidence;
        const sourceAgreement = layer1Result.sourceAgreement;
        const isVolatile = layer1Result.isVolatile;

        // RULE: Close fight lean toward DEC
        // Close fights are harder to finish - apply DEC gravity
        let closeFightDecBoost = 1.0;
        let closeFightFinishPenalty = 1.0;
        if (confidence < this.CLOSE_FIGHT_THRESHOLD) {
            // Scale the DEC boost based on how close the fight is
            // At 50% confidence → max boost; at threshold → mild boost
            // v9: Division-specific DEC cap - decision-heavy divisions get stronger DEC gravity
            const closeness = (this.CLOSE_FIGHT_THRESHOLD - confidence) / (this.CLOSE_FIGHT_THRESHOLD - 50);
            const maxDecBoost = this.DIVISION_CLOSE_FIGHT_DEC_CAP[fight.weightClass] || 1.50;
            closeFightDecBoost = 1.0 + (closeness * (maxDecBoost - 1.0));
            const maxFinishPenalty = (maxDecBoost - 1.0) * 0.5; // Scale finish penalty proportionally
            closeFightFinishPenalty = 1.0 - (closeness * maxFinishPenalty);

            reasoning.push({
                layer: 2,
                type: 'close_fight_rule',
                text: `Close fight detected (${confidence.toFixed(1)}% < ${this.CLOSE_FIGHT_THRESHOLD}%) [${fight.weightClass} cap: ${maxDecBoost}x] - DEC boost ${closeFightDecBoost.toFixed(2)}x, finish penalty ${closeFightFinishPenalty.toFixed(2)}x`
            });

            // If sources also disagree, force DEC
            if (sourceAgreement && sourceAgreement.highConfidenceDisagreement) {
                reasoning.push({
                    layer: 2,
                    type: 'disagreement_rule',
                    text: `High-confidence source disagreement in close fight - forcing DEC prediction`
                });
                return { method: 'DEC', koProb: 0, subProb: 0, forcedByDisagreement: true };
            }
        }

        // Get method distribution from Tapology (using nested structure)
        const tapologyKO = winnerData.tapology?.koTko || 0;
        const tapologySub = winnerData.tapology?.sub || 0;
        const tapologyDec = winnerData.tapology?.dec || 0;
        const hasTapologyMethod = tapologyKO > 0 || tapologySub > 0 || tapologyDec > 0;

        // Get UFC Stats if available (career finish rates)
        const koWinPct = winnerData.ufcStats?.koWinPct || 0;
        const subWinPct = winnerData.ufcStats?.subWinPct || 0;
        const decWinPct = winnerData.ufcStats?.decWinPct || 0;
        const totalFinishPct = koWinPct + subWinPct;
        const opponentFinishLossPct = loserData.ufcStats?.finishLossPct || 0;
        const hasUfcStats = koWinPct > 0 || subWinPct > 0 || opponentFinishLossPct > 0;

        // Get loser's defensive vulnerabilities
        const loserStrDef = loserData.ufcStats?.strDef || 50;
        const loserTdDef = loserData.ufcStats?.tdDef || 50;

        let method = 'DEC';
        let methodReason = '';
        let finalKoProb = 0;
        let finalSubProb = 0;

        // STRATEGY A: Blend Tapology method bars + UFCStats career data + BFO method props + base rate prior
        // v4: Three-way blend with Bayesian anchoring to prevent finish over-prediction
        // v14: BFO method props added as fourth source when available
        const hasBFOMethodData = winnerData.bfo?.methodKO > 0;
        if (hasTapologyMethod || hasUfcStats || hasBFOMethodData) {
            // Start with base probabilities
            let baseKO = 0, baseSub = 0, baseDec = 0;
            let tapologyWeight = 0, ufcStatsWeight = 0;

            // Add Tapology contribution (community prediction) - weight reduced from 0.50 to 0.40
            if (hasTapologyMethod) {
                reasoning.push({
                    layer: 2,
                    type: 'tapology_method',
                    text: `Tapology method prediction for ${winnerData.name}: KO ${tapologyKO}%, SUB ${tapologySub}%, DEC ${tapologyDec}%`
                });
                const tWeight = 0.40;
                baseKO += tapologyKO * tWeight;
                baseSub += tapologySub * tWeight;
                baseDec += tapologyDec * tWeight;
                tapologyWeight = tWeight;
            }

            // Add UFCStats contribution (actual career track record) - with small sample regression
            if (hasUfcStats) {
                reasoning.push({
                    layer: 2,
                    type: 'ufcstats_method',
                    text: `UFCStats career for ${winnerData.name}: KO ${koWinPct.toFixed(0)}%, SUB ${subWinPct.toFixed(0)}%, DEC ${decWinPct.toFixed(0)}%`
                });

                // Detect small sample and regress toward base rates
                const regressed = this.regressSmallSample(koWinPct, subWinPct, decWinPct);
                const effectiveKO = regressed.ko;
                const effectiveSub = regressed.sub;
                const effectiveDec = regressed.dec;

                if (regressed.isSmallSample) {
                    reasoning.push({
                        layer: 2,
                        type: 'small_sample',
                        text: `Small sample detected: regressing UFCStats toward base rates (KO ${effectiveKO.toFixed(0)}%, SUB ${effectiveSub.toFixed(0)}%, DEC ${effectiveDec.toFixed(0)}%)`
                    });
                }

                // UFCStats weight: reduced when small sample, increased when Tapology missing
                let ufcWeight;
                if (hasTapologyMethod) {
                    ufcWeight = regressed.isSmallSample ? 0.15 : 0.30;
                } else {
                    ufcWeight = regressed.isSmallSample ? 0.50 : 0.70;
                }
                baseKO += effectiveKO * ufcWeight;
                baseSub += effectiveSub * ufcWeight;
                baseDec += effectiveDec * ufcWeight;
                ufcStatsWeight = ufcWeight;

                // Bonus: Opponent vulnerability adjustments
                if (opponentFinishLossPct > 60) {
                    // Opponent gets finished a lot - boost finish probability
                    const vulnerabilityBoost = 1.15;
                    baseKO *= vulnerabilityBoost;
                    baseSub *= vulnerabilityBoost;
                    reasoning.push({
                        layer: 2,
                        type: 'vulnerability_boost',
                        text: `Opponent vulnerability: ${loserData.name} has ${opponentFinishLossPct.toFixed(0)}% finish losses - boosting finish probability`
                    });
                }

                // Striker vs poor chin - boost KO
                if (loserStrDef < 50 && koWinPct > 50) {
                    baseKO *= 1.1;
                    reasoning.push({
                        layer: 2,
                        type: 'matchup_boost',
                        text: `Striking matchup: ${loserData.name} has poor striking defense (${loserStrDef}%) vs KO artist - boosting KO`
                    });
                }

                // Wrestler vs poor TD defense - boost SUB potential
                if (loserTdDef < 50 && subWinPct > 30) {
                    baseSub *= 1.1;
                    reasoning.push({
                        layer: 2,
                        type: 'matchup_boost',
                        text: `Grappling matchup: ${loserData.name} has poor TD defense (${loserTdDef}%) vs grappler - boosting SUB`
                    });
                }
            }

            // v14: BFO method prop contribution (market-implied method distribution)
            // When available, this is the single strongest method signal — encodes style,
            // durability, matchup, and recent form via sportsbook pricing
            let bfoMethodWeight = 0;
            const hasBFOMethod = winnerData.bfo?.methodKO > 0;
            if (hasBFOMethod) {
                bfoMethodWeight = 0.30;
                baseKO += winnerData.bfo.methodKO * bfoMethodWeight;
                baseSub += winnerData.bfo.methodSub * bfoMethodWeight;
                baseDec += winnerData.bfo.methodDec * bfoMethodWeight;
                // Reduce Tapology and prior weights when BFO method data available
                if (tapologyWeight > 0) {
                    const reduction = 0.15; // Tapology 0.40→0.25
                    baseKO -= tapologyKO * reduction;
                    baseSub -= tapologySub * reduction;
                    baseDec -= tapologyDec * reduction;
                    tapologyWeight -= reduction;
                }
                reasoning.push({
                    layer: 2,
                    type: 'bfo_method',
                    text: `BFO method props for ${winnerData.name}: KO ${winnerData.bfo.methodKO.toFixed(1)}%, SUB ${winnerData.bfo.methodSub.toFixed(1)}%, DEC ${winnerData.bfo.methodDec.toFixed(1)}%`
                });
            }

            // Add base rate prior (historical UFC method distribution)
            // Acts as Bayesian anchor - prevents finish over-prediction from noisy sources
            // v14: Reduced from 0.30 to 0.20 when BFO method data is available
            const priorWeight = hasBFOMethod ? 0.20 : this.METHOD_PRIOR_WEIGHT;
            baseKO += this.METHOD_BASE_RATE.ko * priorWeight;
            baseSub += this.METHOD_BASE_RATE.sub * priorWeight;
            baseDec += this.METHOD_BASE_RATE.dec * priorWeight;

            // Normalize by total weight
            const totalWeight = tapologyWeight + ufcStatsWeight + bfoMethodWeight + priorWeight;
            if (totalWeight > 0) {
                baseKO = baseKO / totalWeight;
                baseSub = baseSub / totalWeight;
                baseDec = baseDec / totalWeight;
            }

            // Apply weight class bias
            const weightClassBias = this.WEIGHT_CLASS_FINISH_BIAS[fight.weightClass] || { ko: 1, sub: 1, dec: 1 };

            // Calculate adjusted method probabilities
            let koProb = baseKO * weightClassBias.ko * closeFightFinishPenalty;
            let subProb = baseSub * weightClassBias.sub * closeFightFinishPenalty;
            let decProb = baseDec * weightClassBias.dec * closeFightDecBoost;

            // RULE: Lopsided favorite boost for finishes
            // When confidence is very high AND sources all agree, boost finish probability
            // v14: Track whether this fires — mutually exclusive with HC DEC regression
            let lopsidedBoostApplied = false;
            if (confidence >= this.LOPSIDED_FAVORITE_THRESHOLD && sourceAgreement && sourceAgreement.allAgree) {
                const finishBoost = 1.12;
                koProb *= finishBoost;
                subProb *= finishBoost;
                lopsidedBoostApplied = true;
                reasoning.push({
                    layer: 2,
                    type: 'lopsided_favorite',
                    text: `Lopsided favorite rule: ${confidence.toFixed(1)}% confidence + unanimous sources → boosting finish probability by 12%`
                });
            } else if (confidence >= this.LOPSIDED_FAVORITE_THRESHOLD) {
                // High confidence but some disagreement - smaller boost
                const finishBoost = 1.05;
                koProb *= finishBoost;
                subProb *= finishBoost;
                lopsidedBoostApplied = true;
                reasoning.push({
                    layer: 2,
                    type: 'lopsided_favorite',
                    text: `Strong favorite rule: ${confidence.toFixed(1)}% confidence → slight finish boost`
                });
            }

            // RULE: Betting odds finish modifier
            // Big favorites finish fights more often - use betting data we already have
            const winnerBettingPct = winnerData.fightmatrix?.bettingWinPct || null;
            if (winnerBettingPct !== null) {
                let bettingFinishMult = 1.0;
                if (winnerBettingPct >= 80) {
                    bettingFinishMult = 1.18;
                } else if (winnerBettingPct >= 75) {
                    bettingFinishMult = 1.12;
                } else if (winnerBettingPct >= 70) {
                    bettingFinishMult = 1.08;
                }
                if (bettingFinishMult > 1.0) {
                    koProb *= bettingFinishMult;
                    subProb *= bettingFinishMult;
                    reasoning.push({
                        layer: 2,
                        type: 'betting_finish_modifier',
                        text: `Betting odds finish modifier: ${winnerData.name} at ${winnerBettingPct.toFixed(1)}% betting favorite → ${((bettingFinishMult - 1) * 100).toFixed(0)}% finish probability boost`
                    });
                }
            }

            // Apply grappler-specific rules (if UFC stats available)
            if (hasUfcStats) {
                const grapplerAdjustment = this.applyGrapplerRules(winnerData, loserData, fight.weightClass, reasoning);
                koProb *= grapplerAdjustment.koMult;
                subProb *= grapplerAdjustment.subMult;
                decProb *= grapplerAdjustment.decMult;

                // Apply striker-specific rules
                const strikerAdjustment = this.applyStrikerRules(winnerData, loserData, layer1Result, fight.weightClass, reasoning);
                koProb *= strikerAdjustment.koMult;

                // SApM Damage Absorbed modifier: chin vulnerability
                // If the predicted winner absorbs a lot of strikes AND the loser has KO power, boost loser KO upset risk
                // But since we're predicting winner's method, if LOSER has high SApM and WINNER has KO power → boost KO
                const loserSApM = loserData?.ufcStats?.sapm || 0;
                const winnerKOPct = winnerData?.ufcStats?.koWinPct || 0;
                if (loserSApM > 4.0 && winnerKOPct > 50) {
                    koProb *= 1.15;
                    reasoning.push({
                        layer: 2,
                        type: 'damage_absorbed',
                        text: `Chin vulnerability: ${loserData.name} absorbs ${loserSApM.toFixed(1)} SApM vs ${winnerData.name}'s ${winnerKOPct.toFixed(0)}% KO rate → boosting KO`
                    });
                }

                // TD Defense Gate: "Striker's Advantage"
                // If loser's only path is grappling but winner has elite TD defense → favor standing fight
                const winnerTdDef = winnerData?.ufcStats?.tdDef || 50;
                const loserTdAvg = loserData?.ufcStats?.tdAvg || 0;
                const loserSubWinPct = loserData?.ufcStats?.subWinPct || 0;
                const loserDecWinPct = loserData?.ufcStats?.decWinPct || 0;
                const loserIsGrappler = (loserTdAvg >= 2.0 && loserSubWinPct >= 30) || loserSubWinPct >= 50;
                if (winnerTdDef >= 85 && loserIsGrappler) {
                    // Winner keeps it standing → boost KO, reduce SUB
                    koProb *= 1.15;
                    subProb *= 0.75;
                    reasoning.push({
                        layer: 2,
                        type: 'td_defense_gate',
                        text: `Striker's Advantage: ${winnerData.name} has ${winnerTdDef}% TD defense vs grappler ${loserData.name} → fight stays standing`
                    });
                }
            }

            // v11: Finish specialist boost
            // When Tapology strongly predicts a finish method AND fighter is a betting favorite,
            // boost that method to overcome the DEC premium. The DEC premium is correct for
            // marginal cases, but when a strong source signal + market signal align on a finish,
            // the model should trust it rather than defaulting to DEC.
            // Historical: 5 KO→DEC misses (15pts lost) + 2 SUB→DEC misses (6pts lost) in backtest.
            // v14: SUB specialist threshold lowered 55→40%, tiers: ≥65%→1.20x, ≥50%→1.15x, ≥40%→1.10x
            // Betting gate lowered 65→55% (SUB specialists are often underdogs)
            if (tapologySub >= 40 && (winnerBettingPct === null || winnerBettingPct >= 55)) {
                const subBoost = tapologySub >= 65 ? 1.20 : tapologySub >= 50 ? 1.15 : 1.10;
                subProb *= subBoost;
                reasoning.push({
                    layer: 2,
                    type: 'sub_specialist_boost',
                    text: `SUB specialist: Tapology SUB ${tapologySub}%${winnerBettingPct ? ` + ${winnerBettingPct.toFixed(0)}% betting fav` : ''} → ${((subBoost - 1) * 100).toFixed(0)}% SUB boost`
                });
            }
            if (tapologyKO >= 60 && (winnerBettingPct === null || winnerBettingPct >= 65)) {
                const koBoost = tapologyKO >= 75 ? 1.20 : 1.12;
                koProb *= koBoost;
                reasoning.push({
                    layer: 2,
                    type: 'ko_specialist_boost',
                    text: `KO specialist: Tapology KO ${tapologyKO}%${winnerBettingPct ? ` + ${winnerBettingPct.toFixed(0)}% betting fav` : ''} → ${((koBoost - 1) * 100).toFixed(0)}% KO boost`
                });
            }

            // Apply event type modifiers
            const eventModifier = this.applyEventTypeModifier(eventType, fight.isMainEvent, fight.numRounds, reasoning);
            decProb *= eventModifier.decMult;
            koProb *= eventModifier.finishMult;
            subProb *= eventModifier.finishMult;

            // v12: Division-specific finish penalty (BW/FLW)
            // These divisions predict KO at ~38% but actual KO rate is much lower.
            // Apply a blanket finish dampener before EV comparison.
            const divFinishPenalty = this.DIVISION_FINISH_PENALTY[fight.weightClass];
            if (divFinishPenalty) {
                koProb *= divFinishPenalty;
                subProb *= divFinishPenalty;
                reasoning.push({
                    layer: 2,
                    type: 'division_finish_penalty',
                    text: `${fight.weightClass} finish penalty: KO/SUB × ${divFinishPenalty} (decision-heavy division, historically low KO prediction accuracy)`
                });
            }

            // v12: High-confidence DEC regression
            // When winner confidence is high (≥80%) in non-HW/LHW divisions, dominant favorites
            // tend to cruise to decisions rather than chase finishes. Apply additional DEC gravity
            // to counteract compounding finish multipliers.
            // v14: Now mutually exclusive with lopsided favorite finish boost — if sources
            // unanimously agree on a strong favorite, trust the finish signal instead
            const isHeavy = fight.weightClass === 'HW' || fight.weightClass === 'LHW';
            if (confidence >= this.HC_DEC_REGRESSION_THRESHOLD && !isHeavy && !lopsidedBoostApplied) {
                decProb *= this.HC_DEC_REGRESSION_MULTIPLIER;
                reasoning.push({
                    layer: 2,
                    type: 'hc_dec_regression',
                    text: `High-confidence DEC regression: ${confidence.toFixed(1)}% confidence in ${fight.weightClass} → DEC × ${this.HC_DEC_REGRESSION_MULTIPLIER} (dominant favorites cruise to decisions)`
                });
            }

            // Normalize and select method
            const total = koProb + subProb + decProb;
            if (total > 0) {
                koProb = (koProb / total) * 100;
                subProb = (subProb / total) * 100;
                decProb = (decProb / total) * 100;

                // Capture actual probabilities for Layer 3 round prediction
                finalKoProb = koProb;
                finalSubProb = subProb;

                reasoning.push({
                    layer: 2,
                    type: 'adjusted_probs',
                    text: `Adjusted method probabilities: KO ${koProb.toFixed(1)}%, SUB ${subProb.toFixed(1)}%, DEC ${decProb.toFixed(1)}%`
                });

                // v5: Direct EV comparison with DEC certainty premium
                // DEC predictions capture method + round reliably (round is auto-correct)
                // Finish predictions have ~25% round accuracy → higher variance
                // DEC gets a multiplier reflecting this structural advantage
                // v6: Volatile fights get additional DEC boost — uncertain matchups go to
                // decision more often (67.4% volatile accuracy vs 81% non-volatile)
                const volatileBoost = 1.0; // v14: removed volatile DEC boost — 67% of fights flagged volatile, was a permanent DEC tax
                const effectiveMultiplier = this.DEC_EV_MULTIPLIER * volatileBoost;
                const evDec = decProb * effectiveMultiplier;
                const evKo = koProb;
                const evSub = subProb;
                const bestFinishEv = Math.max(evKo, evSub);
                const bestFinishMethod = evKo >= evSub ? 'KO' : 'SUB';

                if (isVolatile) {
                    reasoning.push({
                        layer: 2,
                        type: 'volatile_dec_boost',
                        text: `Volatile fight: DEC multiplier boosted ${this.DEC_EV_MULTIPLIER} → ${effectiveMultiplier.toFixed(2)} (+5% volatile premium)`
                    });
                }

                reasoning.push({
                    layer: 2,
                    type: 'ev_comparison',
                    text: `Method EV: DEC ${evDec.toFixed(1)} (${decProb.toFixed(1)}% × ${effectiveMultiplier.toFixed(2)}), KO ${evKo.toFixed(1)} (${koProb.toFixed(1)}%), SUB ${evSub.toFixed(1)} (${subProb.toFixed(1)}%)`
                });

                if (evDec >= bestFinishEv) {
                    method = 'DEC';
                    methodReason = `DEC selected: EV ${evDec.toFixed(1)} >= best finish ${bestFinishMethod} ${bestFinishEv.toFixed(1)} (DEC ${decProb.toFixed(1)}% × ${effectiveMultiplier.toFixed(2)} certainty premium${isVolatile ? ' [volatile]' : ''})`;
                } else if (evKo >= evSub) {
                    method = 'KO';
                    methodReason = `KO selected: EV ${evKo.toFixed(1)} > DEC ${evDec.toFixed(1)} (KO ${koProb.toFixed(1)}% overcomes DEC ${effectiveMultiplier.toFixed(2)}x premium)`;
                } else {
                    method = 'SUB';
                    methodReason = `SUB selected: EV ${evSub.toFixed(1)} > DEC ${evDec.toFixed(1)} (SUB ${subProb.toFixed(1)}% overcomes DEC ${effectiveMultiplier.toFixed(2)}x premium)`;
                }
            }
        }
        // STRATEGY B: Fall back to UFC Stats finish thresholding if no Tapology method data
        else if (hasUfcStats) {
            reasoning.push({
                layer: 2,
                type: 'finish_rate',
                text: `${winnerData.name} career finishes: ${totalFinishPct.toFixed(1)}% (KO: ${koWinPct}%, SUB: ${subWinPct}%). ${loserData.name} finish losses: ${opponentFinishLossPct}%`
            });

            // Finish Thresholding Rule
            const canPredictFinish = totalFinishPct >= this.FINISH_THRESHOLD &&
                                      opponentFinishLossPct >= this.OPPONENT_FINISH_LOSS_THRESHOLD;

            if (canPredictFinish) {
                // Determine KO vs SUB based on winner's stats
                if (koWinPct > subWinPct) {
                    method = 'KO';
                    finalKoProb = koWinPct;
                    finalSubProb = subWinPct;
                    methodReason = `KO selected: ${winnerData.name} has ${koWinPct}% KO wins vs ${subWinPct}% SUB wins`;
                } else if (subWinPct > koWinPct) {
                    method = 'SUB';
                    finalKoProb = koWinPct;
                    finalSubProb = subWinPct;
                    methodReason = `SUB selected: ${winnerData.name} has ${subWinPct}% SUB wins vs ${koWinPct}% KO wins`;
                } else {
                    // Default to KO if equal
                    method = 'KO';
                    finalKoProb = koWinPct;
                    finalSubProb = subWinPct;
                    methodReason = `KO selected: Equal finish rates, defaulting to KO`;
                }
            } else {
                methodReason = `Finish threshold not met: ${winnerData.name} ${totalFinishPct.toFixed(1)}% finishes (need ${this.FINISH_THRESHOLD}%), ${loserData.name} ${opponentFinishLossPct}% finish losses (need ${this.OPPONENT_FINISH_LOSS_THRESHOLD}%)`;
                reasoning.push({
                    layer: 2,
                    type: 'threshold',
                    text: methodReason
                });
            }
        }
        // STRATEGY C: No method data available - default to DEC
        else {
            methodReason = 'No method data available (no Tapology method bars or UFC Stats) - defaulting to DEC';
            reasoning.push({
                layer: 2,
                type: 'no_data',
                text: methodReason
            });
        }

        reasoning.push({
            layer: 2,
            type: 'result',
            text: `Method: ${method} - ${methodReason}`
        });

        return {
            method,
            koProb: finalKoProb,
            subProb: finalSubProb
        };
    }

    /**
     * Layer 3: Round Prediction
     * Uses continuous scoring with division-specific thresholds to produce
     * natural round variance instead of collapsing all finishes into R1.
     */
    layer3RoundPrediction(fight, layer1Result, layer2Result, reasoning) {
        const winner = layer1Result.winner;
        const loserKey = winner === 'fighterA' ? 'fighterB' : 'fighterA';
        const loserData = fight[loserKey];
        const confidence = layer1Result.confidence;

        const method = layer2Result.method;
        const numRounds = fight.numRounds || 3;
        const isFiveRounder = numRounds === 5;
        const weightClass = fight.weightClass || '';

        let round = 'DEC';

        // If method is decision, round is N/A (full fight)
        if (method === 'DEC') {
            reasoning.push({
                layer: 3,
                type: 'result',
                text: `Round: DEC (Decision - full ${numRounds} rounds)`
            });
            return { round: 'DEC' };
        }

        // Determine dominant method probability from Layer 2
        const dominantPct = method === 'KO' ? layer2Result.koProb : layer2Result.subProb;

        // Calculate early finish profile using continuous scoring
        const winnerData = fight[winner];
        const earlyFinishProfile = this.calculateEarlyFinishProfile(
            method, dominantPct, loserData, confidence, winnerData
        );

        reasoning.push({
            layer: 3,
            type: 'early_finish',
            text: `Early finish profile: ${earlyFinishProfile.score.toFixed(1)} (${earlyFinishProfile.reason})`
        });

        // Select division-specific thresholds
        const thresholds = isFiveRounder
            ? (this.DIVISION_ROUND_THRESHOLDS_5RD[weightClass] || this.FALLBACK_THRESHOLDS_5RD)
            : (this.DIVISION_ROUND_THRESHOLDS_3RD[weightClass] || this.FALLBACK_THRESHOLDS_3RD);

        // Select round based on tiered thresholds (iterate in order: R1, R2, R3...)
        for (const [roundName, threshold] of Object.entries(thresholds)) {
            if (earlyFinishProfile.score >= threshold) {
                round = roundName;
                break;
            }
        }

        // Fallback if no threshold matched
        if (round === 'DEC') {
            round = isFiveRounder ? 'R4' : 'R3';
        }

        const thresholdStr = Object.entries(thresholds)
            .map(([r, t]) => `${r}>=${t}`)
            .join(', ');
        reasoning.push({
            layer: 3,
            type: 'round_selection',
            text: `${weightClass} thresholds [${thresholdStr}] → ${round} (score ${earlyFinishProfile.score.toFixed(1)})`
        });

        reasoning.push({
            layer: 3,
            type: 'result',
            text: `Round: ${round}`
        });

        return { round };
    }

    /**
     * Gather source data for composite calculation
     * Updated to use nested data structure with expanded FightMatrix data
     */
    gatherSourceData(fight) {
        const fighterA = fight.fighterA || {};
        const fighterB = fight.fighterB || {};

        return {
            // Tapology data
            tapologyA: fighterA.tapology?.consensus || 50,
            tapologyB: fighterB.tapology?.consensus || 50,
            tapologyKOA: fighterA.tapology?.koTko || 0,
            tapologySubA: fighterA.tapology?.sub || 0,
            tapologyDecA: fighterA.tapology?.dec || 0,
            tapologyKOB: fighterB.tapology?.koTko || 0,
            tapologySubB: fighterB.tapology?.sub || 0,
            tapologyDecB: fighterB.tapology?.dec || 0,
            // BFO data (replaces DRatings for new events)
            bfoWinPctA: fighterA.bfo?.winPct || null,
            bfoWinPctB: fighterB.bfo?.winPct || null,
            // DRatings data - backward compat for old events
            dratingsA: this.extractDRatingsWinPct(fighterA.dratings),
            dratingsB: this.extractDRatingsWinPct(fighterB.dratings),
            // Legacy CIRRS (backwards compatible)
            fightMatrixA: fighterA.fightMatrix?.cirrs || fighterA.cirrs || null,
            fightMatrixB: fighterB.fightMatrix?.cirrs || fighterB.cirrs || null,
            // Expanded FightMatrix rating systems
            eloK170A: fighterA.fightmatrix?.eloK170 || null,
            eloK170B: fighterB.fightmatrix?.eloK170 || null,
            eloModA: fighterA.fightmatrix?.eloMod || null,
            eloModB: fighterB.fightmatrix?.eloMod || null,
            glickoA: fighterA.fightmatrix?.glicko || null,
            glickoB: fighterB.fightmatrix?.glicko || null,
            whrA: fighterA.fightmatrix?.whr || null,
            whrB: fighterB.fightmatrix?.whr || null,
            // Betting odds
            bettingWinPctA: fighterA.fightmatrix?.bettingWinPct || null,
            bettingWinPctB: fighterB.fightmatrix?.bettingWinPct || null,
            // Age and activity data
            ageA: fighterA.fightmatrix?.age || null,
            ageB: fighterB.fightmatrix?.age || null,
            daysSinceLastFightA: fighterA.fightmatrix?.daysSinceLastFight || null,
            daysSinceLastFightB: fighterB.fightmatrix?.daysSinceLastFight || null,
            // Recent form
            last3RecordA: fighterA.fightmatrix?.last3Record || null,
            last3RecordB: fighterB.fightmatrix?.last3Record || null,
            // UFC Stats striking/grappling (for Layer 1 striking efficiency modifier)
            slpmA: fighterA.ufcStats?.slpm || null,
            sapmA: fighterA.ufcStats?.sapm || null,
            slpmB: fighterB.ufcStats?.slpm || null,
            sapmB: fighterB.ufcStats?.sapm || null,
            strDefA: fighterA.ufcStats?.strDef || null,
            strDefB: fighterB.ufcStats?.strDef || null,
            tdDefA: fighterA.ufcStats?.tdDef || null,
            tdDefB: fighterB.ufcStats?.tdDef || null
        };
    }

    /**
     * Infer missing fighter data from opponent's matchup-specific diff values.
     * FightMatrix diffs encode the matchup: opponent_winPct = 100 - fighter_winPct,
     * opponent_rating = fighter_rating - fighter_diff.
     * Also infers DRatings (matchup-specific) and CIRRS from elo rating.
     */
    inferMissingFromOpponent(fight) {
        const sides = [
            { fighter: 'fighterA', opponent: 'fighterB' },
            { fighter: 'fighterB', opponent: 'fighterA' }
        ];

        for (const { fighter: fKey, opponent: oKey } of sides) {
            const fighter = fight[fKey];
            const opponent = fight[oKey];
            if (!fighter || !opponent) continue;

            const oFm = opponent.fightmatrix;
            if (!oFm) continue;

            // Infer expanded fightmatrix elo ratings from opponent's diffs
            if (!fighter.fightmatrix) fighter.fightmatrix = {};
            const fFm = fighter.fightmatrix;

            for (const system of ['eloK170', 'eloMod', 'glicko', 'whr']) {
                if (!fFm[system] && oFm[system] && oFm[system].winPct != null) {
                    fFm[system] = {
                        rating: Math.round((oFm[system].rating - oFm[system].diff) * 100) / 100,
                        diff: -oFm[system].diff,
                        winPct: Math.round((100 - oFm[system].winPct) * 100) / 100
                    };
                }
            }

            // Infer betting win % from opponent
            if (fFm.bettingWinPct == null && oFm.bettingWinPct != null) {
                fFm.bettingWinPct = Math.round((100 - oFm.bettingWinPct) * 100) / 100;
            }

            // Infer CIRRS from eloK170 rating (CIRRS ≈ eloK170 rating)
            if (!fighter.fightMatrix) fighter.fightMatrix = {};
            if (fighter.fightMatrix.cirrs == null && !fighter.cirrs && fFm.eloK170?.rating) {
                fighter.fightMatrix.cirrs = Math.round(fFm.eloK170.rating);
            }

            // Infer DRatings from opponent (matchup-specific win %)
            const oDratings = this.extractDRatingsWinPct(opponent.dratings);
            const fDratings = this.extractDRatingsWinPct(fighter.dratings);
            if (fDratings === 50 && oDratings !== 50) {
                // Fighter has no dratings (defaulted to 50) but opponent does
                if (!fighter.dratings || (typeof fighter.dratings === 'object' && fighter.dratings.winPct == null) ||
                    fighter.dratings === null || fighter.dratings === undefined) {
                    fighter.dratings = { winPct: Math.round((100 - oDratings) * 100) / 100 };
                }
            }

            // Log inference
            if (fFm.eloK170 && !fight[fKey]._inferred) {
                console.log(`[Prediction] Inferred ${fight[fKey].name} data from opponent ${fight[oKey].name}`);
                fight[fKey]._inferred = true;
            }
        }
    }

    /**
     * Extract DRatings win percentage from various data formats
     * Handles: number, object with winPct, string, null/undefined
     * @returns {number} Win percentage (0-100) or 50 if no valid data
     */
    extractDRatingsWinPct(dratings) {
        // Direct number
        if (typeof dratings === 'number' && !isNaN(dratings)) {
            return dratings;
        }
        // Object with winPct property
        if (dratings && typeof dratings === 'object') {
            const winPct = dratings.winPct;
            if (typeof winPct === 'number' && !isNaN(winPct)) {
                return winPct;
            }
            if (typeof winPct === 'string') {
                const parsed = parseFloat(winPct);
                if (!isNaN(parsed)) {
                    return parsed;
                }
            }
        }
        // String number
        if (typeof dratings === 'string') {
            const parsed = parseFloat(dratings);
            if (!isNaN(parsed)) {
                return parsed;
            }
        }
        // Default when no valid data
        return 50;
    }

    /**
     * Calculate composite win probability from all sources
     * Updated to incorporate multiple FightMatrix rating systems and modifiers
     */
    calculateCompositeWinProb(sources, weightClass) {
        let totalWeight = 0;
        let weightedSumA = 0;
        let primarySourceA = 'composite';
        let primarySourceB = 'composite';
        let maxContributionA = 0;
        const contributions = [];

        // v9: Division-specific source weights (fall back to defaults for unlisted divisions)
        const divWeights = this.DIVISION_SOURCE_WEIGHTS[weightClass] || {};
        const getWeight = (source, defaultWeight) => divWeights[source] !== undefined ? divWeights[source] : defaultWeight;

        // Tapology contribution (default 12%, varies by division)
        const tapologyWeight = getWeight('tapology', 0.12);
        if (sources.tapologyA !== 50 || sources.tapologyB !== 50) {
            weightedSumA += sources.tapologyA * tapologyWeight;
            totalWeight += tapologyWeight;
            contributions.push({ source: 'tapology', value: sources.tapologyA, weight: tapologyWeight });
            if (sources.tapologyA * tapologyWeight > maxContributionA) {
                maxContributionA = sources.tapologyA * tapologyWeight;
                primarySourceA = 'tapology';
            }
        }

        // BFO moneyline contribution (replaces DRatings, default 12%)
        const bfoWeight = getWeight('bfo', 0.12);
        if (sources.bfoWinPctA !== null && sources.bfoWinPctB !== null) {
            weightedSumA += sources.bfoWinPctA * bfoWeight;
            totalWeight += bfoWeight;
            contributions.push({ source: 'bfo', value: sources.bfoWinPctA, weight: bfoWeight });
            if (sources.bfoWinPctA * bfoWeight > maxContributionA) {
                maxContributionA = sources.bfoWinPctA * bfoWeight;
                primarySourceA = 'bfo';
            }
        } else if (sources.dratingsA !== 50 || sources.dratingsB !== 50) {
            // Fallback to DRatings for old events without BFO data
            const dratingsWeight = getWeight('dratings', 0.12);
            weightedSumA += sources.dratingsA * dratingsWeight;
            totalWeight += dratingsWeight;
            contributions.push({ source: 'dratings', value: sources.dratingsA, weight: dratingsWeight });
            if (sources.dratingsA * dratingsWeight > maxContributionA) {
                maxContributionA = sources.dratingsA * dratingsWeight;
                primarySourceA = 'dratings';
            }
        }

        // FightMatrix Betting Odds (default 22% - market signal is strongest predictor)
        if (sources.bettingWinPctA !== null && sources.bettingWinPctB !== null) {
            const bettingWeight = getWeight('betting', 0.22);
            weightedSumA += sources.bettingWinPctA * bettingWeight;
            totalWeight += bettingWeight;
            contributions.push({ source: 'betting', value: sources.bettingWinPctA, weight: bettingWeight });
            if (sources.bettingWinPctA * bettingWeight > maxContributionA) {
                maxContributionA = sources.bettingWinPctA * bettingWeight;
                primarySourceA = 'betting';
            }
        }

        // FightMatrix Elo K170 (default 18% - high predictive validity, encodes opponent quality)
        if (sources.eloK170A !== null && sources.eloK170B !== null) {
            const eloWeight = getWeight('eloK170', 0.18);
            weightedSumA += sources.eloK170A.winPct * eloWeight;
            totalWeight += eloWeight;
            contributions.push({ source: 'eloK170', value: sources.eloK170A.winPct, weight: eloWeight });
            if (sources.eloK170A.winPct * eloWeight > maxContributionA) {
                maxContributionA = sources.eloK170A.winPct * eloWeight;
                primarySourceA = 'eloK170';
            }
        }

        // FightMatrix Elo Modified (default 12% - accounts for recency/momentum)
        if (sources.eloModA !== null && sources.eloModB !== null) {
            const eloModWeight = getWeight('eloMod', 0.12);
            weightedSumA += sources.eloModA.winPct * eloModWeight;
            totalWeight += eloModWeight;
            contributions.push({ source: 'eloMod', value: sources.eloModA.winPct, weight: eloModWeight });
        }

        // FightMatrix Glicko-1 (default 12% - recursive opponent quality encoding)
        if (sources.glickoA !== null && sources.glickoB !== null) {
            const glickoWeight = getWeight('glicko', 0.12);
            weightedSumA += sources.glickoA.winPct * glickoWeight;
            totalWeight += glickoWeight;
            contributions.push({ source: 'glicko', value: sources.glickoA.winPct, weight: glickoWeight });
        }

        // FightMatrix WHR (default 12% - whole-history rating, can be volatile but captures career arc)
        if (sources.whrA !== null && sources.whrB !== null) {
            const whrWeight = getWeight('whr', 0.12);
            weightedSumA += sources.whrA.winPct * whrWeight;
            totalWeight += whrWeight;
            contributions.push({ source: 'whr', value: sources.whrA.winPct, weight: whrWeight });
        }

        // Legacy fallback: use CIRRS if no expanded data
        if (totalWeight === 0 && sources.fightMatrixA !== null && sources.fightMatrixB !== null) {
            const fmWeight = 0.30;
            const ratingGap = sources.fightMatrixA - sources.fightMatrixB;
            const fmWinProbA = 50 + (Math.tanh(ratingGap / 200) * 50);
            weightedSumA += fmWinProbA * fmWeight;
            totalWeight += fmWeight;
            contributions.push({ source: 'cirrs', value: fmWinProbA, weight: fmWeight });
        }

        // Calculate base probability
        let winProbA = totalWeight > 0 ? weightedSumA / totalWeight : 50;

        // Apply modifiers based on age, activity, form, and striking efficiency
        const ageModifier = this.calculateAgeModifier(sources, weightClass);
        const activityModifier = this.calculateActivityModifier(sources);
        const formModifier = this.calculateFormModifier(sources);
        const strikingModifier = this.calculateStrikingEfficiencyModifier(sources);

        // Apply modifiers (small adjustments, capped)
        winProbA += ageModifier + activityModifier + formModifier + strikingModifier;
        winProbA = Math.max(5, Math.min(95, winProbA)); // Cap between 5-95%

        // v9: Division-confidence calibration
        // Shrink confidence toward 50% for historically weak divisions
        // This makes the model less decisive in divisions where its edge is thinner
        const calibrationFactor = this.DIVISION_CONFIDENCE_CALIBRATION[weightClass] || 0.95;
        if (calibrationFactor < 1.0) {
            const deviation = winProbA - 50;
            winProbA = 50 + (deviation * calibrationFactor);
        }

        const winProbB = 100 - winProbA;

        // Determine primary source for B
        if (winProbB > winProbA) {
            if (sources.tapologyB > sources.tapologyA) primarySourceB = 'tapology';
            else if (sources.bfoWinPctB !== null && sources.bfoWinPctB > sources.bfoWinPctA) primarySourceB = 'bfo';
            else if (sources.dratingsB > sources.dratingsA) primarySourceB = 'dratings';
            else if (sources.bettingWinPctB > sources.bettingWinPctA) primarySourceB = 'betting';
            else if (sources.eloK170B?.winPct > sources.eloK170A?.winPct) primarySourceB = 'eloK170';
        }

        return {
            winProbA,
            winProbB,
            primarySourceA,
            primarySourceB,
            contributions,
            modifiers: { age: ageModifier, activity: activityModifier, form: formModifier, striking: strikingModifier }
        };
    }

    /**
     * Calculate age-based modifier with non-linear "Age Wall" penalty
     * MMA performance cliff at 35 is steep, especially in lighter weight classes.
     * Age gaps of 7+ years are significant. HW gets reduced penalty (power offsets age).
     */
    calculateAgeModifier(sources, weightClass) {
        if (sources.ageA === null || sources.ageB === null) return 0;

        const ageDiff = sources.ageB - sources.ageA; // positive = A is younger
        let modifier = 0;

        // Weight class scaling: lighter classes penalize age more, HW less
        const AGE_WEIGHT_CLASS_SCALE = {
            'FLW': 1.3, 'BW': 1.2, 'FW': 1.1, 'LW': 1.1,
            'WW': 1.0, 'MW': 1.0, 'LHW': 0.8, 'HW': 0.6,
            'WFLW': 1.2, 'WBW': 1.1, 'WFW': 1.0, 'WSW': 1.2
        };
        const wcScale = AGE_WEIGHT_CLASS_SCALE[weightClass] || 1.0;

        // Non-linear age cliff: penalty accelerates past 35
        const applyAgeCliff = (age) => {
            if (age < 35) return 0;
            if (age < 37) return -1.5;  // 35-36: mild decline
            if (age < 39) return -3.5;  // 37-38: steep decline
            return -5.5;                // 39+: severe decline
        };

        modifier += applyAgeCliff(sources.ageA) * wcScale;
        modifier -= applyAgeCliff(sources.ageB) * wcScale; // Benefit to A if B is old

        // Large age gap bonus: 7+ year gap is a significant factor
        const absAgeDiff = Math.abs(ageDiff);
        if (absAgeDiff >= 8) {
            modifier += Math.sign(ageDiff) * 3.0 * wcScale;
        } else if (absAgeDiff >= 7) {
            modifier += Math.sign(ageDiff) * 2.0 * wcScale;
        }

        // Cap total age modifier
        return Math.max(-8, Math.min(8, modifier));
    }

    /**
     * Calculate activity/ring rust modifier
     * 365+ days = automatic 10% confidence reduction per the "Year Off" rule
     * Graduated penalties below that threshold
     */
    calculateActivityModifier(sources) {
        if (sources.daysSinceLastFightA === null || sources.daysSinceLastFightB === null) return 0;

        let modifier = 0;

        const applyRingRust = (days) => {
            if (days > 730) return -10;    // 2+ years = extreme rust
            if (days > 545) return -8;     // 18+ months
            if (days > 365) return -6;     // 1+ year = "Year Off" penalty
            if (days > this.LAYOFF_MODERATE) return -2; // 300-365 days
            return 0;
        };

        // Ring rust penalty for A
        modifier += applyRingRust(sources.daysSinceLastFightA);
        // Ring rust penalty for B (benefit to A)
        modifier -= applyRingRust(sources.daysSinceLastFightB);

        return modifier;
    }

    /**
     * Calculate source agreement score
     * Returns how many sources agree on the same winner and agreement strength
     */
    calculateSourceAgreement(sources) {
        const picks = [];

        // Collect all source picks with their confidence levels
        if (sources.tapologyA !== 50 || sources.tapologyB !== 50) {
            picks.push({
                source: 'Tapology',
                picksA: sources.tapologyA > 50,
                confidence: Math.abs(sources.tapologyA - 50)
            });
        }

        if (sources.dratingsA !== 50 || sources.dratingsB !== 50) {
            picks.push({
                source: 'DRatings',
                picksA: sources.dratingsA > 50,
                confidence: Math.abs(sources.dratingsA - 50)
            });
        }

        if (sources.bettingWinPctA !== null) {
            picks.push({
                source: 'Betting',
                picksA: sources.bettingWinPctA > 50,
                confidence: Math.abs(sources.bettingWinPctA - 50)
            });
        }

        if (sources.eloK170A !== null) {
            picks.push({
                source: 'EloK170',
                picksA: sources.eloK170A.winPct > 50,
                confidence: Math.abs(sources.eloK170A.winPct - 50)
            });
        }

        if (sources.eloModA !== null) {
            picks.push({
                source: 'EloMod',
                picksA: sources.eloModA.winPct > 50,
                confidence: Math.abs(sources.eloModA.winPct - 50)
            });
        }

        if (sources.glickoA !== null) {
            picks.push({
                source: 'Glicko',
                picksA: sources.glickoA.winPct > 50,
                confidence: Math.abs(sources.glickoA.winPct - 50)
            });
        }

        if (sources.whrA !== null) {
            picks.push({
                source: 'WHR',
                picksA: sources.whrA.winPct > 50,
                confidence: Math.abs(sources.whrA.winPct - 50)
            });
        }

        // Legacy CIRRS
        if (sources.fightMatrixA !== null && sources.fightMatrixB !== null && !sources.eloK170A) {
            picks.push({
                source: 'CIRRS',
                picksA: sources.fightMatrixA > sources.fightMatrixB,
                confidence: Math.abs(sources.fightMatrixA - sources.fightMatrixB) / 10 // Normalize
            });
        }

        if (picks.length === 0) {
            return { agreementCount: 0, totalSources: 0, allAgree: false, disagreingSources: [] };
        }

        // Count how many pick A vs B
        const picksACount = picks.filter(p => p.picksA).length;
        const picksBCount = picks.length - picksACount;
        const majorityPicksA = picksACount > picksBCount;
        const agreementCount = majorityPicksA ? picksACount : picksBCount;

        // Find disagreeing sources
        const disagreingSources = picks
            .filter(p => p.picksA !== majorityPicksA)
            .map(p => p.source);

        // Check if high-confidence sources disagree
        // v11: raised threshold from 15→25 (a source at 35% = 15% confidence is barely leaning)
        const highConfidenceDisagreement = picks
            .filter(p => p.picksA !== majorityPicksA && p.confidence > 25)
            .length > 0;

        return {
            agreementCount,
            totalSources: picks.length,
            allAgree: disagreingSources.length === 0,
            disagreingSources,
            majorityPicksA,
            highConfidenceDisagreement,
            agreementRatio: agreementCount / picks.length
        };
    }

    /**
     * Calculate recent form modifier based on last 3 fights
     */
    calculateFormModifier(sources) {
        if (sources.last3RecordA === null || sources.last3RecordB === null) return 0;

        const parseRecord = (record) => {
            const parts = record.split('-').map(n => parseInt(n));
            return { wins: parts[0] || 0, losses: parts[1] || 0, draws: parts[2] || 0 };
        };

        const recordA = parseRecord(sources.last3RecordA);
        const recordB = parseRecord(sources.last3RecordB);

        let modifier = 0;

        // Perfect recent form bonus
        if (recordA.wins === 3 && recordA.losses === 0) modifier += 1.5;
        if (recordB.wins === 3 && recordB.losses === 0) modifier -= 1.5;

        // Recent losing form penalty
        if (recordA.losses >= 2) modifier -= 1.5;
        if (recordB.losses >= 2) modifier += 1.5;

        return modifier;
    }

    /**
     * v7: Calculate striking efficiency modifier for Layer 1 winner selection.
     * Net striking = SLpM - SApM (positive = lands more than absorbs).
     * A large differential between fighters is a strong predictive signal
     * that currently has zero input into winner probability.
     *
     * Positive modifier favors Fighter A, negative favors Fighter B.
     * Capped at ±4 points to prevent overriding strong source consensus.
     */
    calculateStrikingEfficiencyModifier(sources) {
        const slpmA = sources.slpmA;
        const sapmA = sources.sapmA;
        const slpmB = sources.slpmB;
        const sapmB = sources.sapmB;

        // Need both fighters' striking data
        if (slpmA === null || sapmA === null || slpmB === null || sapmB === null) return 0;

        // Net striking advantage for each fighter
        const netA = slpmA - sapmA;  // positive = outstrikes opponents
        const netB = slpmB - sapmB;

        // Differential: positive favors A
        const differential = netA - netB;

        let modifier = 0;

        // Tiered modifier based on differential magnitude
        const absDiff = Math.abs(differential);
        if (absDiff >= 3.0) {
            modifier = 3.5;  // Large gap — one fighter clearly outstrikes
        } else if (absDiff >= 2.0) {
            modifier = 2.5;  // Significant advantage
        } else if (absDiff >= 1.0) {
            modifier = 1.5;  // Moderate edge
        } else {
            return 0;  // < 1.0 differential — not meaningful
        }

        // Apply direction (positive = favor A, negative = favor B)
        if (differential < 0) modifier = -modifier;

        // TD defense bonus: if the better striker also has great TD defense,
        // they can keep the fight standing — amplify the striking advantage
        const strDefA = sources.strDefA;
        const strDefB = sources.strDefB;
        const tdDefA = sources.tdDefA;
        const tdDefB = sources.tdDefB;

        if (modifier > 0 && tdDefA !== null && tdDefA >= 80) {
            modifier *= 1.15;  // Better striker can stuff takedowns → amplify
        } else if (modifier < 0 && tdDefB !== null && tdDefB >= 80) {
            modifier *= 1.15;
        }

        // Cap at ±4 to prevent overriding strong source consensus
        return Math.max(-4, Math.min(4, modifier));
    }

    /**
     * Detect small UFCStats sample and regress method percentages toward base rates.
     * When a fighter has very few UFC wins (indicated by extreme distributions like 100%/0%),
     * their method splits are unreliable. Apply Bayesian shrinkage toward historical averages.
     */
    regressSmallSample(koWinPct, subWinPct, decWinPct) {
        const zeroCount = [koWinPct, subWinPct, decWinPct].filter(v => v === 0).length;
        const hundredCount = [koWinPct, subWinPct, decWinPct].filter(v => v >= 100).length;
        const isSmallSample = hundredCount >= 1 || zeroCount >= 2;

        if (!isSmallSample) {
            return { ko: koWinPct, sub: subWinPct, dec: decWinPct, isSmallSample: false };
        }

        // Bayesian shrinkage: blend 60% actual + 40% base rate
        const shrinkage = 0.40;
        return {
            ko: koWinPct * (1 - shrinkage) + this.METHOD_BASE_RATE.ko * shrinkage,
            sub: subWinPct * (1 - shrinkage) + this.METHOD_BASE_RATE.sub * shrinkage,
            dec: decWinPct * (1 - shrinkage) + this.METHOD_BASE_RATE.dec * shrinkage,
            isSmallSample: true
        };
    }

    /**
     * Check if sources disagree on the winner
     * Updated to check multiple FightMatrix rating systems
     */
    checkSourceDisagreement(sources) {
        const disagreements = [];
        const picks = [];

        // Tapology pick
        const tapologyPicksA = sources.tapologyA > 50;
        picks.push({ source: 'Tapology', picksA: tapologyPicksA, value: sources.tapologyA });

        // BFO pick (preferred over DRatings)
        if (sources.bfoWinPctA !== null) {
            picks.push({ source: 'BFO', picksA: sources.bfoWinPctA > 50, value: sources.bfoWinPctA });
        } else {
            // Fallback to DRatings for old events
            const dratingsPicksA = sources.dratingsA > 50;
            picks.push({ source: 'DRatings', picksA: dratingsPicksA, value: sources.dratingsA });
        }

        // Betting odds pick
        if (sources.bettingWinPctA !== null) {
            const bettingPicksA = sources.bettingWinPctA > 50;
            picks.push({ source: 'Betting', picksA: bettingPicksA, value: sources.bettingWinPctA });
        }

        // Elo K170 pick
        if (sources.eloK170A !== null) {
            const eloPicksA = sources.eloK170A.winPct > 50;
            picks.push({ source: 'Elo K170', picksA: eloPicksA, value: sources.eloK170A.winPct });
        }

        // Glicko pick
        if (sources.glickoA !== null) {
            const glickoPicksA = sources.glickoA.winPct > 50;
            picks.push({ source: 'Glicko', picksA: glickoPicksA, value: sources.glickoA.winPct });
        }

        // WHR pick (often disagrees with others)
        if (sources.whrA !== null) {
            const whrPicksA = sources.whrA.winPct > 50;
            picks.push({ source: 'WHR', picksA: whrPicksA, value: sources.whrA.winPct });
        }

        // v11: Tightened volatility trigger
        // Previously: ANY single dissenting source triggered volatile (12/14 fights flagged)
        // Now: require 2+ dissenters OR 1 strong dissenter (>60% confidence in opposite direction)
        const majorityPicksA = picks.filter(p => p.picksA).length > picks.length / 2;
        const dissenting = picks.filter(p => p.picksA !== majorityPicksA);
        const strongDissenters = dissenting.filter(d => Math.abs(d.value - 50) > 10);

        // Only flag as disagreement if meaningful dissent exists
        if (dissenting.length >= 2 || strongDissenters.length >= 1) {
            dissenting.forEach(d => {
                disagreements.push(`${d.source} picks ${d.picksA ? 'Fighter A' : 'Fighter B'} (${d.value.toFixed(1)}%)`);
            });
        }

        // Check for close margins (within 5%) — v14: tightened from 2+ to 3+ sources
        // With 67% of fights flagged volatile, the 2-source threshold was too loose
        const closeMargins = picks.filter(p => Math.abs(p.value - 50) < 5);
        if (closeMargins.length >= 3) {
            disagreements.push(`Close margins: ${closeMargins.map(p => p.source).join(', ')}`);
        }

        // Check for high variance between sources (raised from 25 to 30)
        const values = picks.map(p => p.value);
        const maxValue = Math.max(...values);
        const minValue = Math.min(...values);
        if (maxValue - minValue > 30) {
            disagreements.push(`High variance between sources (${minValue.toFixed(1)}% - ${maxValue.toFixed(1)}%)`);
        }

        return disagreements.length > 0 ? disagreements.join('; ') : null;
    }

    /**
     * Apply grappler-specific rules
     * Updated to use nested ufcStats structure
     */
    applyGrapplerRules(winnerData, loserData, weightClass, reasoning) {
        let koMult = 1;
        let subMult = 1;
        let decMult = 1;

        const tdAvg = winnerData?.ufcStats?.tdAvg || 0;
        const subWinPct = winnerData?.ufcStats?.subWinPct || 0;
        const ctrlTime = winnerData?.ufcStats?.ctrlTime || 0;

        // Wrestler-to-SUB bias: high TDs + high SUB wins
        if (tdAvg >= this.WRESTLER_TD_THRESHOLD && subWinPct >= this.WRESTLER_SUB_WIN_THRESHOLD) {
            subMult = 1.3;
            koMult = 0.8;
            reasoning.push({
                layer: 2,
                type: 'grappler_rule',
                text: `Wrestler-to-SUB bias: ${winnerData.name} has ${tdAvg} TDs/15min and ${subWinPct}% SUB wins - boosting SUB probability`
            });
        }

        // Veteran Control Bias: high TDs + high control time = Decision favored
        if (tdAvg >= this.VETERAN_CONTROL_TD_THRESHOLD && ctrlTime >= this.VETERAN_CONTROL_TIME_THRESHOLD) {
            decMult = 1.25;
            koMult *= 0.9;
            subMult *= 0.9;
            reasoning.push({
                layer: 2,
                type: 'grappler_rule',
                text: `Veteran Control Bias: ${winnerData.name} has ${tdAvg} TDs/15min and ${ctrlTime} min control/round - favoring DEC`
            });
        }

        // Underdog Grappler Caution: check if loser has grappling threat
        const loserTdAvg = loserData?.ufcStats?.tdAvg || 0;
        const loserSubWinPct = loserData?.ufcStats?.subWinPct || 0;
        if (loserTdAvg >= 2.0 && loserSubWinPct >= 40) {
            // Reduce finish confidence when opponent has grappling threat
            koMult *= 0.9;
            subMult *= 0.9;
            reasoning.push({
                layer: 2,
                type: 'grappler_caution',
                text: `Underdog Grappler Caution: ${loserData.name} has grappling threat (${loserTdAvg} TDs, ${loserSubWinPct}% SUB wins) - reducing finish confidence`
            });
        }

        return { koMult, subMult, decMult };
    }

    /**
     * Apply striker-specific rules
     * Updated to use nested ufcStats structure
     */
    applyStrikerRules(winnerData, loserData, layer1Result, weightClass, reasoning) {
        let koMult = 1;

        const koWinPct = winnerData?.ufcStats?.koWinPct || 0;
        const winnerSlpm = winnerData?.ufcStats?.slpm || 0;
        const winnerSapm = winnerData?.ufcStats?.sapm || 0;
        const loserSlpm = loserData?.ufcStats?.slpm || 0;
        const loserSapm = loserData?.ufcStats?.sapm || 0;

        // Early KO Threat Multiplier: underdog with high KO rate
        if (layer1Result.confidence < this.EARLY_KO_THREAT_TAPOLOGY_THRESHOLD &&
            koWinPct >= this.EARLY_KO_KO_WIN_THRESHOLD) {
            koMult = 1.25;
            reasoning.push({
                layer: 2,
                type: 'striker_rule',
                text: `Early KO Threat Multiplier: ${winnerData.name} is underdog (<${this.EARLY_KO_THREAT_TAPOLOGY_THRESHOLD}% confidence) with ${koWinPct}% KO wins - boosting KO probability`
            });
        }

        // Striking Differential: (Landed - Absorbed) is more predictive than raw volume
        // A fighter landing 6/min but absorbing 5 is worse than one landing 4 but absorbing 2
        if (winnerSlpm > 0 && winnerSapm > 0) {
            const winnerDiff = winnerSlpm - winnerSapm;
            const loserDiff = (loserSlpm > 0 && loserSapm > 0) ? (loserSlpm - loserSapm) : 0;
            const netAdvantage = winnerDiff - loserDiff;

            if (netAdvantage >= 3.0) {
                // Dominant striking differential (e.g., +2.5 vs -0.5)
                koMult *= 1.15;
                reasoning.push({
                    layer: 2,
                    type: 'striking_differential',
                    text: `Striking Differential: ${winnerData.name} net ${winnerDiff > 0 ? '+' : ''}${winnerDiff.toFixed(1)} vs ${loserData.name} net ${loserDiff > 0 ? '+' : ''}${loserDiff.toFixed(1)} (advantage: ${netAdvantage > 0 ? '+' : ''}${netAdvantage.toFixed(1)}) → significant KO boost`
                });
            } else if (netAdvantage >= 1.5) {
                // Moderate striking advantage
                koMult *= 1.08;
                reasoning.push({
                    layer: 2,
                    type: 'striking_differential',
                    text: `Striking Differential: ${winnerData.name} net ${winnerDiff > 0 ? '+' : ''}${winnerDiff.toFixed(1)} vs ${loserData.name} net ${loserDiff > 0 ? '+' : ''}${loserDiff.toFixed(1)} (advantage: +${netAdvantage.toFixed(1)}) → moderate KO boost`
                });
            }
        }

        return { koMult };
    }

    /**
     * Apply event type modifiers
     */
    applyEventTypeModifier(eventType, isMainEvent, numRounds, reasoning) {
        let decMult = 1;
        let finishMult = 1;

        // PPV/ABC main events tend toward decisions in close fights
        if ((eventType === 'ppv' || eventType === 'abc') && isMainEvent) {
            decMult = 1.20;
            finishMult = 0.9;
            reasoning.push({
                layer: 2,
                type: 'event_modifier',
                text: `PPV/ABC Main Event modifier: Slightly favoring DEC in high-profile fight`
            });
        }

        // 5-round fights have more time for decisions
        if (numRounds === 5) {
            decMult *= 1.15;
            reasoning.push({
                layer: 2,
                type: 'event_modifier',
                text: `5-round fight modifier: Slightly higher DEC probability due to fight length`
            });
        }

        return { decMult, finishMult };
    }

    /**
     * Calculate early finish profile for round prediction
     * Uses continuous scoring: base = dominant method % × scale factor,
     * plus small capped bonuses. Creates natural variance between fights
     * instead of collapsing everything into R1.
     */
    calculateEarlyFinishProfile(method, dominantPct, loserData, confidence, winnerData) {
        const bonuses = [];
        let totalBonus = 0;

        // Base score: continuous function of how dominant the predicted method is
        const base = dominantPct * this.METHOD_CONFIDENCE_SCALE;

        // Bonus: Loser is a power puncher (both fighters trading = earlier finishes)
        const loserTapologyKO = loserData?.tapology?.koTko || 0;
        if (loserTapologyKO >= 50) {
            bonuses.push(`loser is power puncher (${loserTapologyKO}% KO)`);
            totalBonus += this.BONUS_LOSER_POWER_PUNCHER;
        }

        // Bonus: Lopsided matchup (big favorites finish more often)
        if (confidence >= 73) {
            bonuses.push('lopsided matchup');
            totalBonus += this.BONUS_LOPSIDED;
        }

        // Bonus: Strong betting favorite (odds ~< -300, ~75%+ betting win prob)
        // Research: fights with strong favorites end significantly faster
        const bettingWinPct = winnerData?.fightmatrix?.bettingWinPct || null;
        if (bettingWinPct !== null && bettingWinPct >= 75) {
            const favBonus = bettingWinPct >= 80 ? 5.0 : 3.0;
            bonuses.push(`strong betting fav (${bettingWinPct.toFixed(0)}%)`);
            totalBonus += favBonus;
        }

        const cappedBonus = Math.min(totalBonus, this.MAX_BONUS_CAP);
        const score = base + cappedBonus;

        return {
            score,
            base,
            dominantPct,
            bonuses,
            cappedBonus,
            reason: `base ${base.toFixed(1)} from ${method} ${dominantPct.toFixed(0)}% × ${this.METHOD_CONFIDENCE_SCALE} + bonus ${cappedBonus.toFixed(1)}${bonuses.length > 0 ? ' (' + bonuses.join(', ') + ')' : ''}`
        };
    }
}

// Export singleton instance
const predictionEngine = new PredictionEngine();
